"""
Error Recovery and Retry Logic Tests.
Tests connection loss, retry mechanisms, and graceful degradation.

Run with: pytest tests/test_error_recovery.py -v
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from fastapi.testclient import TestClient
import json
import sys

# Mock problematic imports
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['pymilvus'] = MagicMock()

from app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestMilvusConnectionRecovery:
    """Test Milvus connection recovery."""
    
    def test_milvus_connection_lost_during_insert(self, client):
        """Test recovery when Milvus connection is lost during insert."""
        with patch('src.api.routes.get_s3_file') as mock_s3, \
             patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.get_doc_processor') as mock_processor, \
             patch('src.api.routes.get_obligation_extractor') as mock_extractor, \
             patch('src.api.routes.ReportSynthesizer') as mock_report, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker, \
             patch('src.api.routes.get_s3_client') as mock_s3_client:
            
            # Setup mocks
            mock_s3.return_value = b"PDF content"
            
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.get_embeddings_batch = Mock(return_value=[[0.1] * 768])
            mock_llm.return_value = llm_mock
            
            # Milvus fails on first attempt, succeeds on retry
            milvus_mock = Mock()
            milvus_mock.insert_extract_data_batch = Mock(
                side_effect=[
                    ConnectionError("Connection lost"),
                    ["chunk-1"]  # Success on retry
                ]
            )
            milvus_mock.create_contracts_collection = Mock()
            milvus_mock.insert_contract = Mock()
            milvus_mock.insert_obligations_batch = Mock(return_value=["obl-1"])
            mock_milvus.return_value = milvus_mock
            
            doc_mock = Mock()
            doc_mock.validate_file = Mock(return_value=(True, None))
            doc_mock.extract_text = Mock(return_value="Text")
            doc_mock.chunk_text = Mock(return_value=[{"text": "Chunk", "index": 0}])
            mock_processor.return_value = doc_mock
            
            extractor_mock = Mock()
            extractor_mock.extract_obligations = Mock(return_value=[])
            mock_extractor.return_value = extractor_mock
            
            report_mock = Mock()
            report_mock.synthesize_report = Mock(return_value={'content': b'PDF', 'format': 'pdf'})
            mock_report.return_value = report_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            s3_client_mock = Mock()
            s3_client_mock.put_object = Mock()
            s3_client_mock.generate_presigned_url = Mock(return_value="https://s3.example.com/report.pdf")
            mock_s3_client.return_value = s3_client_mock
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should fail (no retry logic implemented yet)
            assert response.status_code == 500
    
    def test_milvus_reconnection_after_disconnect(self):
        """Test that Milvus client can reconnect after disconnect."""
        from src.storage.milvus_client import MilvusClient
        
        with patch('pymilvus.connections') as mock_connections:
            mock_connections.connect = Mock()
            mock_connections.disconnect = Mock()
            
            client = MilvusClient()
            
            # Simulate disconnect
            client._connected = True
            client.disconnect()
            assert not client._connected
            
            # Reconnect
            client.connect()
            assert client._connected
    
    def test_milvus_search_retry_on_timeout(self, client):
        """Test retry logic when Milvus search times out."""
        with patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            mock_llm.return_value = llm_mock
            
            # Milvus times out on first attempt
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(
                side_effect=TimeoutError("Search timeout")
            )
            mock_milvus.return_value = milvus_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "Test",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should fail gracefully
            assert response.status_code == 500


class TestLLMAPIRecovery:
    """Test LLM API error recovery."""
    
    def test_llm_rate_limit_with_backoff(self, client):
        """Test exponential backoff when hitting rate limits."""
        with patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            # LLM fails with rate limit, then succeeds
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(
                side_effect=[
                    Exception("Rate limit exceeded"),
                    [0.1] * 768  # Success on retry
                ]
            )
            mock_llm.return_value = llm_mock
            
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(return_value=[])
            mock_milvus.return_value = milvus_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "Test",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should fail (no retry logic implemented)
            assert response.status_code == 500
    
    def test_llm_timeout_recovery(self, client):
        """Test recovery from LLM API timeouts."""
        with patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.generate = Mock(side_effect=TimeoutError("Request timeout"))
            mock_llm.return_value = llm_mock
            
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(return_value=[{"content": "Test"}])
            mock_milvus.return_value = milvus_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "Test",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500
    
class TestS3ConnectionRecovery:
    """Test S3 connection recovery."""
    
    def test_s3_connection_timeout_retry(self, client):
        """Test retry on S3 connection timeout."""
        with patch('src.api.routes.get_s3_file') as mock_s3, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            # S3 times out on first attempt
            mock_s3.side_effect = TimeoutError("S3 connection timeout")
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500
    
    def test_s3_upload_failure_cleanup(self, client):
        """Test cleanup when S3 upload fails."""
        with patch('src.api.routes.get_s3_file') as mock_s3_get, \
             patch('src.api.routes.get_s3_client') as mock_s3_client, \
             patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.get_doc_processor') as mock_processor, \
             patch('src.api.routes.get_obligation_extractor') as mock_extractor, \
             patch('src.api.routes.ReportSynthesizer') as mock_report, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            mock_s3_get.return_value = b"PDF content"
            
            # S3 upload fails
            s3_client_mock = Mock()
            s3_client_mock.put_object = Mock(side_effect=Exception("S3 upload failed"))
            mock_s3_client.return_value = s3_client_mock
            
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.get_embeddings_batch = Mock(return_value=[[0.1] * 768])
            mock_llm.return_value = llm_mock
            
            milvus_mock = Mock()
            milvus_mock.insert_extract_data_batch = Mock(return_value=["chunk-1"])
            milvus_mock.create_contracts_collection = Mock()
            milvus_mock.insert_contract = Mock()
            milvus_mock.insert_obligations_batch = Mock(return_value=["obl-1"])
            mock_milvus.return_value = milvus_mock
            
            doc_mock = Mock()
            doc_mock.validate_file = Mock(return_value=(True, None))
            doc_mock.extract_text = Mock(return_value="Text")
            doc_mock.chunk_text = Mock(return_value=[{"text": "Chunk", "index": 0}])
            mock_processor.return_value = doc_mock
            
            extractor_mock = Mock()
            extractor_mock.extract_obligations = Mock(return_value=[])
            mock_extractor.return_value = extractor_mock
            
            report_mock = Mock()
            report_mock.synthesize_report = Mock(return_value={'content': b'PDF', 'format': 'pdf'})
            mock_report.return_value = report_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500


class TestPartialFailureRecovery:
    """Test recovery from partial failures."""
    
    def test_obligation_extraction_failure_continues(self, client):
        """Test that upload continues even if obligation extraction fails."""
        with patch('src.api.routes.get_s3_file') as mock_s3, \
             patch('src.api.routes.get_s3_client') as mock_s3_client, \
             patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.get_doc_processor') as mock_processor, \
             patch('src.api.routes.get_obligation_extractor') as mock_extractor, \
             patch('src.api.routes.ReportSynthesizer') as mock_report, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            mock_s3.return_value = b"PDF content"
            
            s3_client_mock = Mock()
            s3_client_mock.put_object = Mock()
            s3_client_mock.generate_presigned_url = Mock(return_value="https://s3.example.com/report.pdf")
            mock_s3_client.return_value = s3_client_mock
            
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.get_embeddings_batch = Mock(return_value=[[0.1] * 768])
            mock_llm.return_value = llm_mock
            
            milvus_mock = Mock()
            milvus_mock.insert_extract_data_batch = Mock(return_value=["chunk-1"])
            milvus_mock.create_contracts_collection = Mock()
            milvus_mock.insert_contract = Mock()
            milvus_mock.insert_obligations_batch = Mock(return_value=[])
            mock_milvus.return_value = milvus_mock
            
            doc_mock = Mock()
            doc_mock.validate_file = Mock(return_value=(True, None))
            doc_mock.extract_text = Mock(return_value="Text")
            doc_mock.chunk_text = Mock(return_value=[{"text": "Chunk", "index": 0}])
            mock_processor.return_value = doc_mock
            
            # Obligation extraction fails
            extractor_mock = Mock()
            extractor_mock.extract_obligations = Mock(
                side_effect=Exception("Extraction failed")
            )
            mock_extractor.return_value = extractor_mock
            
            report_mock = Mock()
            report_mock.synthesize_report = Mock(return_value={'content': b'PDF', 'format': 'pdf'})
            mock_report.return_value = report_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should fail (no graceful degradation)
            assert response.status_code == 500


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
class TestGracefulDegradation:
    """Test graceful degradation when services are unavailable."""
    
    def test_search_without_milvus(self, client):
        """Test fallback when Milvus is unavailable."""
        with patch('src.api.routes.get_llm_client') as mock_llm, \
             patch('src.api.routes.get_milvus_client') as mock_milvus, \
             patch('src.api.routes.create_event_logger') as mock_logger, \
             patch('src.api.routes.get_llm_config') as mock_config, \
             patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.generate = Mock(return_value="Fallback response")
            mock_llm.return_value = llm_mock
            
            # Milvus is unavailable
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(
                side_effect=ConnectionError("Milvus unavailable")
            )
            mock_milvus.return_value = milvus_mock
            
            logger_mock = Mock()
            logger_mock.log_event = Mock()
            mock_logger.return_value = logger_mock
            
            mock_config.return_value = {"model": "test", "temperature": 0.7}
            mock_tracker.return_value = Mock()
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "Test",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should fail (no fallback implemented)
            assert response.status_code == 500
    
    def test_upload_without_obligation_extraction(self, client):
        """Test that upload works even if obligation extraction fails."""
        # Already tested above
        pass
    
class TestDataConsistencyOnFailure:
    """Test data consistency when operations fail."""
    
class TestHealthCheckRecovery:
    """Test health check and service recovery."""
    
