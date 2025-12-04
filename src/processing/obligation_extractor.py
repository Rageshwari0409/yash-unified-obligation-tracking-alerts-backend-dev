"""
Obligation Extractor for Contract Analysis.
Uses LLM to extract structured obligations with dates from contract documents.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

OBLIGATION_TYPES = [
    "renewal_date",
    "payment_schedule", 
    "service_delivery_deadline",
    "compliance_milestone",
    "termination_notice",
    "performance_review",
    "audit_requirement",
    "reporting_deadline",
    "warranty_expiration",
    "notice_period",
    "milestone_deadline",
    "other"
]

EXTRACTION_PROMPT = """You are an expert contract analyst specializing in obligation extraction and deadline tracking. Your task is to meticulously analyze the contract and extract ALL obligations, commitments, deadlines, and time-sensitive requirements.

## EXTRACTION GUIDELINES:

### What to Extract:
- Payment obligations and schedules
- Delivery deadlines and milestones
- Renewal and termination dates
- Compliance requirements and audit schedules
- Notice periods and notification requirements
- Performance reviews and reporting deadlines
- Warranty periods and expiration dates
- Service level agreements (SLAs) and response times
- Insurance and indemnification requirements
- Confidentiality and non-disclosure periods

### Obligation Categories:
{obligation_types}

### For EACH obligation found, provide:

1. **type**: Select the most appropriate category from the list above. Use "other" only if no category fits.

2. **description**: Write a clear, concise description (max 100 words) that captures:
   - What must be done
   - Any specific requirements or conditions
   - Relevant amounts, quantities, or specifications

3. **due_date**: 
   - Use YYYY-MM-DD format for specific dates (e.g., "2025-12-31")
   - For relative dates, be specific (e.g., "30 days after contract signing", "within 5 business days of invoice receipt")
   - If no specific date, use "ongoing" or "as needed"

4. **party_responsible**: 
   - Identify the party clearly (e.g., "Vendor", "Client", "Service Provider", "Both Parties")
   - If multiple parties, list primary responsible party first

5. **recurrence**: 
   - Choose from: "one-time", "daily", "weekly", "monthly", "quarterly", "annually", "none"
   - Use "one-time" for single occurrences
   - Use "none" for ongoing obligations without fixed schedule

6. **priority**: 
   - **high**: Financial obligations, legal compliance, critical deadlines, termination clauses
   - **medium**: Regular deliverables, standard reporting, routine reviews
   - **low**: Optional items, informational requirements, non-binding suggestions

7. **original_text**: 
   - Quote the exact relevant text from the contract (max 200 words)
   - Include enough context to understand the obligation
   - Use "..." to indicate omitted text if needed

## IMPORTANT RULES:
- Extract EVERY obligation, even if it seems minor
- Do NOT infer obligations that aren't explicitly stated
- If a date is ambiguous, include both possible interpretations
- For recurring obligations, extract the first occurrence and note the recurrence
- Pay special attention to "shall", "must", "will", "required to", "obligated to"
- Look for hidden obligations in definitions, exhibits, and appendices

## CONTRACT TEXT:
{contract_text}

## OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanations, no markdown, no additional text.

If NO obligations are found, return: []

If obligations are found, use this exact structure:
[
  {{
    "type": "payment_schedule",
    "description": "Monthly service fee of $5,000 due on the 15th of each month for the duration of the contract term",
    "due_date": "2025-01-15",
    "party_responsible": "Client",
    "recurrence": "monthly",
    "priority": "high",
    "original_text": "Client shall pay Vendor a monthly service fee of Five Thousand Dollars ($5,000.00) on or before the fifteenth (15th) day of each calendar month"
  }}
]

JSON Output:"""


class ObligationExtractor:
    """Extracts structured obligations from contract text using LLM."""
    
    def __init__(self, llm_client):
        """
        Initialize the obligation extractor.
        
        Args:
            llm_client: LiteLLM client instance for LLM calls
        """
        self.llm_client = llm_client
        self.obligation_types = OBLIGATION_TYPES
        logger.info("ObligationExtractor initialized")
    
    def extract_obligations(
        self,
        contract_text: str,
        contract_id: str,
        token_tracker=None,
        llm_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract obligations from contract text using LLM.
        
        Args:
            contract_text: Full contract text or relevant chunks
            contract_id: ID of the contract document
            token_tracker: Optional token usage tracker
            llm_params: Optional LLM parameters
            
        Returns:
            List of extracted obligation dictionaries
        """
        try:
            # Truncate text if too long (keep first 15000 chars for context)
            text_for_analysis = contract_text[:15000] if len(contract_text) > 15000 else contract_text
            
            prompt = EXTRACTION_PROMPT.format(
                obligation_types=", ".join(self.obligation_types),
                contract_text=text_for_analysis
            )
            
            # Call LLM for extraction
            response = self.llm_client.generate(
                prompt=prompt,
                token_tracker=token_tracker,
                llm_params=llm_params
            )
            
            # Parse JSON response
            obligations = self._parse_llm_response(response)
            
            # Enrich obligations with metadata
            enriched_obligations = []
            for idx, obligation in enumerate(obligations):
                enriched = self._enrich_obligation(obligation, contract_id, idx)
                enriched_obligations.append(enriched)
            
            logger.info(f"Extracted {len(enriched_obligations)} obligations from contract {contract_id}")
            return enriched_obligations
            
        except Exception as e:
            logger.error(f"Error extracting obligations: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract JSON obligations."""
        try:
            # Try to find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return []
    
    def _enrich_obligation(
        self,
        obligation: Dict[str, Any],
        contract_id: str,
        index: int
    ) -> Dict[str, Any]:
        """Add metadata and validate obligation fields."""
        # Generate unique obligation ID
        obligation_id = f"{contract_id}_obl_{index}"
        
        # Normalize and validate fields
        enriched = {
            "id": obligation_id,
            "contract_id": contract_id,
            "type": self._validate_type(obligation.get("type", "other")),
            "description": str(obligation.get("description", ""))[:500],
            "due_date": self._normalize_date(obligation.get("due_date", "")),
            "party_responsible": str(obligation.get("party_responsible", ""))[:200],
            "recurrence": self._validate_recurrence(obligation.get("recurrence", "none")),
            "priority": self._validate_priority(obligation.get("priority", "medium")),
            "original_text": str(obligation.get("original_text", ""))[:1000],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        
        return enriched

    def _validate_type(self, obligation_type: str) -> str:
        """Validate obligation type against allowed types."""
        obligation_type = str(obligation_type).lower().strip()
        if obligation_type in self.obligation_types:
            return obligation_type
        return "other"

    def _validate_recurrence(self, recurrence: str) -> str:
        """Validate recurrence value."""
        valid_recurrences = ["one-time", "daily", "weekly", "monthly", "quarterly", "annually", "none"]
        recurrence = str(recurrence).lower().strip()
        if recurrence in valid_recurrences:
            return recurrence
        return "none"

    def _validate_priority(self, priority: str) -> str:
        """Validate priority value."""
        valid_priorities = ["high", "medium", "low"]
        priority = str(priority).lower().strip()
        if priority in valid_priorities:
            return priority
        return "medium"

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format if possible."""
        if not date_str:
            return ""

        date_str = str(date_str).strip()

        # Already in correct format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str

        # Try common date formats
        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in date_formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Return original if can't parse (might be relative date like "30 days after signing")
        return date_str[:50]
