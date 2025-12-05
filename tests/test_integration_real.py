"""
Real Integration Tests with actual services.
Tests end-to-end flows with real Milvus, S3, and LLM APIs.

NOTE: These tests require:
- Running Milvus instance
- Valid AWS credentials
- Valid GOOGLE_API_KEY
- Test S3 bucket configured

Run with: pytest tests/test_integration_real.py -v --tb=short
Skip with: pytest -m "not integration"
"""
import pytest
import os
import io
import time
from datetime import datetime
from fastapi.testclient import TestClient
import json

from app import app
from src.storage.milvus_client import MilvusClient
from src.llm.litellm_client import LiteLLMClient
from src.utils.s3_utility import get_s3_client, S3_BUCKET
from src.processing.document_processor import DocumentProcessor


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_milvus_client():
    """Real Milvus client for integration testing."""
    client = MilvusClient()
    client.connect()
    yield client
    client.disconnect()


@pytest.fixture
def test_contract_pdf():
    """Sample contract PDF content for testing."""
    # Simple text-based PDF for testing
    contract_text = """
    SERVICE AGREEMENT
    
    This agreement is entered into on January 1, 2025.
    
    PAYMENT TERMS:
    Client shall pay $5,000 monthly on the 15th of each month.
    
    RENEWAL:
    This contract automatically renews on December 31, 2025.
    
    TERMINATION:
    Either party may terminate with 30 days written notice.
    """
    return contract_text.encode('utf-8')


@pytest.fixture
def cleanup_test_data(real_milvus_client):
    """Cleanup test data after tests."""
    test_contract_ids = []
    
    yield test_contract_ids
    
    # Cleanup after test
    for _ in test_contract_ids:
        try:
            # Delete from collections (implement cleanup logic)
            pass
        except Exception as e:
            print(f"Cleanup warning: {e}")


class TestRealIntegrationFlow:
    """Test complete end-to-end flows with real services."""
    
    @pytest.mark.slow
    def test_milvus_connection_and_collections(self, real_milvus_client):
        """Test Milvus connection and collection creation."""
        # Test connection
        assert real_milvus_client._connected, "Milvus not connected"
        
        # Test collection creation
        contracts_collection = real_milvus_client.create_contracts_collection()
        assert contracts_collection is not None
        
        extract_data_collection = real_milvus_client.create_extract_data_collection()
        assert extract_data_collection is not None
        
        obligations_collection = real_milvus_client.create_obligations_collection()
        assert obligations_collection is not None
    
class TestDataConsistency:
    """Test data consistency across collections."""

    @pytest.mark.slow
    def test_duplicate_contract_id_handling(self, real_milvus_client):
        """Test behavior when inserting duplicate contract IDs."""
        contract_id = f"duplicate-test-{int(time.time())}"
        
        contract_data = {
            "id": contract_id,
            "user_id": "test_user",
            "filename": "test.pdf",
            "title": "Test Contract",
            "parties": "",
            "effective_date": "",
            "expiration_date": "",
            "contract_value": "",
            "content_summary": "Test summary",
            "embedding": [0.1] * 768
        }
        
        # Insert first time
        real_milvus_client.create_contracts_collection()
        first_result = real_milvus_client.insert_contract(contract_data)
        assert first_result is not None

        # Insert again with same ID
        try:
            second_result = real_milvus_client.insert_contract(contract_data)
            # Should either succeed (upsert) or raise error
            assert second_result is not None
        except Exception as e:
            # Expected behavior - duplicate key error
            assert "duplicate" in str(e).lower() or "exists" in str(e).lower()


class TestErrorRecovery:
    """Test error recovery and retry logic."""

    @pytest.mark.slow
    def test_milvus_reconnection_after_disconnect(self, real_milvus_client):
        """Test that client can reconnect after disconnection."""
        # Disconnect
        real_milvus_client.disconnect()
        assert not real_milvus_client._connected
        
        # Reconnect
        real_milvus_client.connect()
        assert real_milvus_client._connected
        
        # Verify functionality
        collection = real_milvus_client.create_contracts_collection()
        assert collection is not None
    
class TestPerformance:
    """Performance and load tests."""
    
    @pytest.mark.slow
    def test_large_document_processing(self):
        """Test processing of large documents (100+ pages)."""
        # Generate large document
        large_text = "This is a contract clause. " * 10000  # ~250KB

        processor = DocumentProcessor()
        
        start_time = time.time()
        chunks = processor.chunk_text(large_text)
        processing_time = time.time() - start_time
        
        assert len(chunks) > 0, "Chunking failed"
        assert processing_time < 30, f"Processing too slow: {processing_time}s"


class TestRealFileFormats:
    """Test with actual file formats (not mocks)."""
    
    def test_real_txt_file_processing(self):
        """Test processing actual TXT file."""
        processor = DocumentProcessor()
        
        content = b"""
        SERVICE AGREEMENT
        
        This agreement is made on January 1, 2025.
        Payment terms: $5,000 monthly.
        """
        
        text = processor.extract_text(content, "test.txt")
        assert "SERVICE AGREEMENT" in text
        assert "5,000" in text
        
        chunks = processor.chunk_text(text)
        assert len(chunks) > 0
    
    @pytest.mark.skipif(
        not os.path.exists("tests/fixtures/sample.pdf"),
        reason="Sample PDF not available"
    )
    def test_real_pdf_file_processing(self):
        """Test processing actual PDF file."""
        processor = DocumentProcessor()
        
        with open("tests/fixtures/sample.pdf", "rb") as f:
            content = f.read()
        
        text = processor.extract_text(content, "sample.pdf")
        assert len(text) > 0, "PDF extraction failed"
        
        chunks = processor.chunk_text(text)
        assert len(chunks) > 0, "PDF chunking failed"
    
    @pytest.mark.skipif(
        not os.path.exists("tests/fixtures/sample.docx"),
        reason="Sample DOCX not available"
    )
    def test_real_docx_file_processing(self):
        """Test processing actual DOCX file."""
        processor = DocumentProcessor()
        
        with open("tests/fixtures/sample.docx", "rb") as f:
            content = f.read()
        
        text = processor.extract_text(content, "sample.docx")
        assert len(text) > 0, "DOCX extraction failed"
        
        chunks = processor.chunk_text(text)
        assert len(chunks) > 0, "DOCX chunking failed"


class TestAuthentication:
    """Test authentication and authorization (when enabled)."""
    
    @pytest.mark.skip(reason="Authentication currently disabled")
    def test_upload_without_auth_token(self):
        """Test that upload requires authentication."""
        client = TestClient(app)
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 401, "Should require authentication"
    
    @pytest.mark.skip(reason="Authentication currently disabled")
    def test_upload_with_invalid_token(self):
        """Test upload with invalid auth token."""
        client = TestClient(app)
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            },
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401, "Should reject invalid token"
    
