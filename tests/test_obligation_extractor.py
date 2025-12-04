"""
Unit tests for ObligationExtractor class.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import sys

# Mock problematic imports
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['pymilvus'] = MagicMock()

from src.processing.obligation_extractor import ObligationExtractor, OBLIGATION_TYPES


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    llm_mock = Mock()
    llm_mock.generate = Mock(return_value='[{"type": "payment_schedule", "description": "Monthly payment", "due_date": "2025-01-15", "party_responsible": "Client", "recurrence": "monthly", "priority": "high", "original_text": "Payment due monthly"}]')
    return llm_mock


@pytest.fixture
def extractor(mock_llm_client):
    """Obligation extractor instance."""
    return ObligationExtractor(mock_llm_client)


class TestObligationExtractorInit:
    """Test suite for ObligationExtractor initialization."""
    
    def test_initialization(self, mock_llm_client):
        """Test extractor initialization."""
        extractor = ObligationExtractor(mock_llm_client)
        assert extractor.llm_client == mock_llm_client
        assert extractor.obligation_types == OBLIGATION_TYPES
    
    def test_obligation_types_loaded(self, extractor):
        """Test that obligation types are properly loaded."""
        assert len(extractor.obligation_types) > 0
        assert "payment_schedule" in extractor.obligation_types
        assert "renewal_date" in extractor.obligation_types


class TestExtractObligations:
    """Test suite for extract_obligations method."""
    
    def test_extract_obligations_success(self, extractor, mock_llm_client):
        """Test successful obligation extraction."""
        contract_text = "Payment is due on the 15th of each month."
        contract_id = "doc-2025-001"
        
        obligations = extractor.extract_obligations(contract_text, contract_id)
        
        assert len(obligations) > 0
        assert obligations[0]["contract_id"] == contract_id
        assert obligations[0]["type"] == "payment_schedule"
        assert "id" in obligations[0]
    
    def test_extract_obligations_with_token_tracker(self, extractor, mock_llm_client):
        """Test obligation extraction with token tracker."""
        token_tracker = Mock()
        contract_text = "Payment is due on the 15th of each month."
        contract_id = "doc-2025-001"

        # Call extract_obligations with token tracker
        extractor.extract_obligations(contract_text, contract_id, token_tracker=token_tracker)

        # Verify LLM was called with token tracker
        mock_llm_client.generate.assert_called_once()
        call_kwargs = mock_llm_client.generate.call_args[1]
        assert call_kwargs.get("token_tracker") == token_tracker

    def test_extract_obligations_with_llm_params(self, extractor, mock_llm_client):
        """Test obligation extraction with LLM parameters."""
        llm_params = {"temperature": 0.5, "max_tokens": 1000}
        contract_text = "Payment is due on the 15th of each month."
        contract_id = "doc-2025-001"

        # Call extract_obligations with llm_params
        extractor.extract_obligations(contract_text, contract_id, llm_params=llm_params)

        # Verify LLM was called with params
        call_kwargs = mock_llm_client.generate.call_args[1]
        assert call_kwargs.get("llm_params") == llm_params

    def test_extract_obligations_long_text_truncation(self, extractor, mock_llm_client):
        """Test that long contract text is truncated."""
        long_text = "A" * 20000  # Very long text
        contract_id = "doc-2025-001"

        # Call extract_obligations with long text
        extractor.extract_obligations(long_text, contract_id)

        # Verify LLM was called with truncated text
        call_args = mock_llm_client.generate.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        # Prompt should not contain full 20000 chars
        assert len(prompt) < 20000
    
    def test_extract_obligations_empty_response(self, extractor):
        """Test handling of empty LLM response."""
        extractor.llm_client.generate = Mock(return_value="[]")
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        
        assert obligations == []
    
    def test_extract_obligations_invalid_json(self, extractor):
        """Test handling of invalid JSON response."""
        extractor.llm_client.generate = Mock(return_value="Invalid JSON")
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        
        assert obligations == []
    
    def test_extract_obligations_llm_error(self, extractor):
        """Test handling of LLM errors."""
        extractor.llm_client.generate = Mock(side_effect=Exception("LLM API error"))
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        
        assert obligations == []


class TestParseLLMResponse:
    """Test suite for _parse_llm_response method."""
    
    def test_parse_valid_json_array(self, extractor):
        """Test parsing valid JSON array."""
        response = '[{"type": "payment_schedule", "description": "Monthly payment"}]'
        result = extractor._parse_llm_response(response)
        
        assert len(result) == 1
        assert result[0]["type"] == "payment_schedule"
    
    def test_parse_json_with_surrounding_text(self, extractor):
        """Test parsing JSON with surrounding text."""
        response = 'Here are the obligations: [{"type": "payment_schedule"}] End of response'
        result = extractor._parse_llm_response(response)
        
        assert len(result) == 1
    
    def test_parse_empty_array(self, extractor):
        """Test parsing empty array."""
        response = '[]'
        result = extractor._parse_llm_response(response)
        
        assert result == []
    
    def test_parse_invalid_json(self, extractor):
        """Test parsing invalid JSON."""
        response = 'Not a JSON array'
        result = extractor._parse_llm_response(response)
        
        assert result == []
    
    def test_parse_malformed_json(self, extractor):
        """Test parsing malformed JSON."""
        response = '[{"type": "payment_schedule",}]'  # Trailing comma
        result = extractor._parse_llm_response(response)
        
        assert result == []


class TestEnrichObligation:
    """Test suite for _enrich_obligation method."""
    
    def test_enrich_obligation_basic(self, extractor):
        """Test basic obligation enrichment."""
        obligation = {
            "type": "payment_schedule",
            "description": "Monthly payment",
            "due_date": "2025-01-15",
            "party_responsible": "Client",
            "recurrence": "monthly",
            "priority": "high",
            "original_text": "Payment due monthly"
        }
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert enriched["id"] == "doc-2025-001_obl_0"
        assert enriched["contract_id"] == "doc-2025-001"
        assert enriched["status"] == "pending"
        assert "created_at" in enriched
    
    def test_enrich_obligation_validates_type(self, extractor):
        """Test that obligation type is validated."""
        obligation = {"type": "invalid_type"}
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert enriched["type"] == "other"
    
    def test_enrich_obligation_validates_recurrence(self, extractor):
        """Test that recurrence is validated."""
        obligation = {"recurrence": "invalid_recurrence"}
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert enriched["recurrence"] == "none"
    
    def test_enrich_obligation_validates_priority(self, extractor):
        """Test that priority is validated."""
        obligation = {"priority": "invalid_priority"}
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert enriched["priority"] == "medium"
    
    def test_enrich_obligation_truncates_long_fields(self, extractor):
        """Test that long fields are truncated."""
        obligation = {
            "description": "A" * 1000,
            "party_responsible": "B" * 500,
            "original_text": "C" * 2000
        }
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert len(enriched["description"]) <= 500
        assert len(enriched["party_responsible"]) <= 200
        assert len(enriched["original_text"]) <= 1000


class TestValidateType:
    """Test suite for _validate_type method."""
    
    def test_validate_valid_type(self, extractor):
        """Test validation of valid obligation type."""
        result = extractor._validate_type("payment_schedule")
        assert result == "payment_schedule"
    
    def test_validate_invalid_type(self, extractor):
        """Test validation of invalid obligation type."""
        result = extractor._validate_type("invalid_type")
        assert result == "other"
    
    def test_validate_type_case_insensitive(self, extractor):
        """Test that type validation is case insensitive."""
        result = extractor._validate_type("PAYMENT_SCHEDULE")
        assert result == "payment_schedule"
    
    def test_validate_type_with_whitespace(self, extractor):
        """Test that type validation handles whitespace."""
        result = extractor._validate_type("  payment_schedule  ")
        assert result == "payment_schedule"


class TestValidateRecurrence:
    """Test suite for _validate_recurrence method."""
    
    def test_validate_valid_recurrence(self, extractor):
        """Test validation of valid recurrence."""
        for recurrence in ["one-time", "daily", "weekly", "monthly", "quarterly", "annually", "none"]:
            result = extractor._validate_recurrence(recurrence)
            assert result == recurrence
    
    def test_validate_invalid_recurrence(self, extractor):
        """Test validation of invalid recurrence."""
        result = extractor._validate_recurrence("invalid")
        assert result == "none"
    
    def test_validate_recurrence_case_insensitive(self, extractor):
        """Test that recurrence validation is case insensitive."""
        result = extractor._validate_recurrence("MONTHLY")
        assert result == "monthly"


class TestValidatePriority:
    """Test suite for _validate_priority method."""
    
    def test_validate_valid_priority(self, extractor):
        """Test validation of valid priority."""
        for priority in ["high", "medium", "low"]:
            result = extractor._validate_priority(priority)
            assert result == priority
    
    def test_validate_invalid_priority(self, extractor):
        """Test validation of invalid priority."""
        result = extractor._validate_priority("invalid")
        assert result == "medium"
    
    def test_validate_priority_case_insensitive(self, extractor):
        """Test that priority validation is case insensitive."""
        result = extractor._validate_priority("HIGH")
        assert result == "high"


class TestNormalizeDate:
    """Test suite for _normalize_date method."""
    
    def test_normalize_already_formatted_date(self, extractor):
        """Test normalization of already formatted date."""
        result = extractor._normalize_date("2025-01-15")
        assert result == "2025-01-15"
    
    def test_normalize_slash_date_dmy(self, extractor):
        """Test normalization of DD/MM/YYYY format."""
        result = extractor._normalize_date("15/01/2025")
        assert result == "2025-01-15"
    
    def test_normalize_slash_date_mdy(self, extractor):
        """Test normalization of MM/DD/YYYY format."""
        result = extractor._normalize_date("01/15/2025")
        assert result == "2025-01-15"
    
    def test_normalize_text_date(self, extractor):
        """Test normalization of text date format."""
        result = extractor._normalize_date("January 15, 2025")
        assert result == "2025-01-15"
    
    def test_normalize_relative_date(self, extractor):
        """Test that relative dates are preserved."""
        result = extractor._normalize_date("30 days after signing")
        assert result == "30 days after signing"
    
    def test_normalize_empty_date(self, extractor):
        """Test normalization of empty date."""
        result = extractor._normalize_date("")
        assert result == ""
    
    def test_normalize_invalid_date(self, extractor):
        """Test normalization of invalid date."""
        result = extractor._normalize_date("not a date")
        assert result == "not a date"
    
    def test_normalize_date_truncation(self, extractor):
        """Test that very long date strings are truncated."""
        long_date = "A" * 100
        result = extractor._normalize_date(long_date)
        assert len(result) <= 50



class TestObligationExtractorNegative:
    """Negative test cases for ObligationExtractor."""
    
    def test_extract_with_none_contract_text(self, extractor):
        """Test extraction with None contract text."""
        try:
            obligations = extractor.extract_obligations(None, "doc-2025-001")
            # Should handle gracefully or return empty
            assert isinstance(obligations, list)
        except (TypeError, AttributeError):
            # Expected to raise error
            pass
    
    def test_extract_with_empty_contract_id(self, extractor, mock_llm_client):
        """Test extraction with empty contract ID."""
        obligations = extractor.extract_obligations("Contract text", "")
        assert isinstance(obligations, list)
    
    def test_extract_with_malformed_json_response(self, extractor):
        """Test extraction when LLM returns malformed JSON."""
        extractor.llm_client.generate = Mock(return_value='{"type": "payment", "incomplete')
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        assert obligations == []
    
    def test_extract_with_json_array_in_text(self, extractor):
        """Test extraction when JSON is embedded in other text."""
        extractor.llm_client.generate = Mock(
            return_value='Here are the obligations: [{"type": "payment_schedule"}] as requested.'
        )
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        assert len(obligations) == 1
    
    def test_extract_with_nested_json(self, extractor):
        """Test extraction with nested JSON structures."""
        extractor.llm_client.generate = Mock(
            return_value='[{"type": "payment_schedule", "nested": {"field": "value"}}]'
        )
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        assert len(obligations) > 0
    
    def test_extract_with_llm_timeout(self, extractor):
        """Test extraction when LLM times out."""
        extractor.llm_client.generate = Mock(side_effect=TimeoutError("LLM timeout"))
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        assert obligations == []
    
    def test_extract_with_llm_connection_error(self, extractor):
        """Test extraction when LLM connection fails."""
        extractor.llm_client.generate = Mock(side_effect=ConnectionError("Connection failed"))
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        assert obligations == []
    
    def test_extract_with_unicode_contract_text(self, extractor, mock_llm_client):
        """Test extraction with unicode characters in contract."""
        contract_text = "契約条件：毎月15日に支払い。Paiement le 15 de chaque mois."
        
        obligations = extractor.extract_obligations(contract_text, "doc-2025-001")
        assert isinstance(obligations, list)
    
    def test_extract_with_special_characters(self, extractor, mock_llm_client):
        """Test extraction with special characters."""
        contract_text = "Payment: $5,000.00 @ 15th of month (net-30) [clause 4.2]"
        
        obligations = extractor.extract_obligations(contract_text, "doc-2025-001")
        assert isinstance(obligations, list)
    
    def test_extract_with_html_content(self, extractor, mock_llm_client):
        """Test extraction with HTML content in contract."""
        contract_text = "<p>Payment due on <b>15th</b> of each month.</p><script>alert('xss')</script>"
        
        obligations = extractor.extract_obligations(contract_text, "doc-2025-001")
        assert isinstance(obligations, list)
    
    def test_enrich_with_missing_fields(self, extractor):
        """Test enrichment with missing required fields."""
        obligation = {}  # Empty obligation
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert "id" in enriched
        assert "contract_id" in enriched
        assert enriched["type"] == "other"  # Default value
    
    def test_enrich_with_null_values(self, extractor):
        """Test enrichment with null values."""
        obligation = {
            "type": None,
            "description": None,
            "due_date": None
        }
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        assert enriched["type"] == "other"
        assert enriched["description"] == "None"
    
    def test_enrich_with_numeric_strings(self, extractor):
        """Test enrichment with numeric strings."""
        obligation = {
            "type": 123,
            "description": 456,
            "priority": 789
        }
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 0)
        
        # Should convert to strings
        assert isinstance(enriched["description"], str)
    
    def test_validate_type_with_none(self, extractor):
        """Test type validation with None."""
        result = extractor._validate_type(None)
        assert result == "other"
    
    def test_validate_type_with_number(self, extractor):
        """Test type validation with number."""
        result = extractor._validate_type(123)
        assert result == "other"
    
    def test_normalize_date_with_invalid_formats(self, extractor):
        """Test date normalization with various invalid formats."""
        invalid_dates = [
            "32/13/2025",  # Invalid day/month
            "2025-13-45",  # Invalid month/day
            "not a date at all",
            "00/00/0000",
            "99-99-9999"
        ]

        for date_str in invalid_dates:
            result = extractor._normalize_date(date_str)
            # Should return original or truncated string
            assert isinstance(result, str)

    def test_normalize_date_with_none(self, extractor):
        """Test date normalization with None."""
        result = extractor._normalize_date(None)
        assert result == ""
    
    def test_normalize_date_with_future_year(self, extractor):
        """Test date normalization with far future year."""
        result = extractor._normalize_date("2099-12-31")
        assert result == "2099-12-31"
    
    def test_normalize_date_with_past_year(self, extractor):
        """Test date normalization with past year."""
        result = extractor._normalize_date("1900-01-01")
        assert result == "1900-01-01"
    
    def test_parse_llm_response_with_multiple_arrays(self, extractor):
        """Test parsing when response contains multiple JSON arrays."""
        response = '[{"type": "payment"}] and also [{"type": "renewal"}]'
        result = extractor._parse_llm_response(response)

        # May return empty list if it can't parse properly, or parse first array
        assert isinstance(result, list)
    
    def test_parse_llm_response_with_escaped_characters(self, extractor):
        """Test parsing JSON with escaped characters."""
        response = r'[{"description": "Payment \"due\" on 15th"}]'
        result = extractor._parse_llm_response(response)
        
        assert len(result) == 1
    
    def test_extract_with_extremely_long_text(self, extractor, mock_llm_client):
        """Test extraction with extremely long contract text."""
        long_text = "Contract clause. " * 100000  # Very long text
        
        obligations = extractor.extract_obligations(long_text, "doc-2025-001")
        
        # Should truncate and still work
        assert isinstance(obligations, list)
        # Verify truncation happened
        call_args = mock_llm_client.generate.call_args
        if call_args:
            prompt = call_args[1]["prompt"]
            assert len(prompt) < len(long_text)
    
    def test_extract_with_no_obligations_found(self, extractor):
        """Test extraction when no obligations are found."""
        extractor.llm_client.generate = Mock(return_value='[]')
        
        obligations = extractor.extract_obligations(
            "This is a simple letter with no obligations.",
            "doc-2025-001"
        )
        
        assert obligations == []
    
    def test_extract_with_duplicate_obligations(self, extractor):
        """Test extraction with duplicate obligations in response."""
        extractor.llm_client.generate = Mock(
            return_value='[{"type": "payment_schedule"}, {"type": "payment_schedule"}]'
        )
        
        obligations = extractor.extract_obligations("Contract text", "doc-2025-001")
        
        # Should return all, even duplicates (each gets unique ID)
        assert len(obligations) == 2
        assert obligations[0]["id"] != obligations[1]["id"]


class TestValidationEdgeCases:
    """Edge case tests for validation methods."""
    
    def test_validate_recurrence_with_mixed_case(self, extractor):
        """Test recurrence validation with mixed case."""
        result = extractor._validate_recurrence("MoNtHlY")
        assert result == "monthly"
    
    def test_validate_priority_with_extra_whitespace(self, extractor):
        """Test priority validation with extra whitespace."""
        result = extractor._validate_priority("  high  ")
        assert result == "high"
    
    def test_validate_type_with_underscore_variations(self, extractor):
        """Test type validation with underscore variations."""
        result = extractor._validate_type("payment__schedule")
        assert result == "other"
    
    def test_enrich_obligation_with_very_long_index(self, extractor):
        """Test enrichment with very large index number."""
        obligation = {"type": "payment_schedule"}

        enriched = extractor._enrich_obligation(obligation, "doc-2025-001", 999999)

        assert "999999" in enriched["id"]
    
    def test_enrich_obligation_with_special_chars_in_contract_id(self, extractor):
        """Test enrichment with special characters in contract ID."""
        obligation = {"type": "payment_schedule"}
        
        enriched = extractor._enrich_obligation(obligation, "doc-2025-001_special!@#", 0)
        
        assert enriched["contract_id"] == "doc-2025-001_special!@#"
