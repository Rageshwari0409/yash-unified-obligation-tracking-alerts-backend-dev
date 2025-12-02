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

EXTRACTION_PROMPT = """You are an expert contract analyst. Extract ALL obligations, deadlines, and important dates from the following contract text.

For each obligation found, provide:
1. type: One of these categories: {obligation_types}
2. description: Brief description of the obligation
3. due_date: The date in YYYY-MM-DD format (if specific date mentioned), or relative date like "30 days after signing"
4. party_responsible: Who is responsible for this obligation
5. recurrence: "one-time", "daily", "weekly", "monthly", "quarterly", "annually", or "none"
6. priority: "high", "medium", or "low" based on importance
7. original_text: The exact text from the contract mentioning this obligation

CONTRACT TEXT:
{contract_text}

Return ONLY a valid JSON array of obligations. If no obligations found, return an empty array [].
Example format:
[
  {{
    "type": "payment_schedule",
    "description": "Monthly service fee payment",
    "due_date": "2025-01-15",
    "party_responsible": "Client",
    "recurrence": "monthly",
    "priority": "high",
    "original_text": "Client shall pay $5000 monthly on the 15th of each month"
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
