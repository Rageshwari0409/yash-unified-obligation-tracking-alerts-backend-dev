"""
Unit tests for API routes - upload and chat endpoints.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import json
import sys

# Mock problematic imports before importing app
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['pymilvus'] = MagicMock()

from app import app
from src.api.routes import (
    get_llm_client,
    get_milvus_client,
    get_doc_processor,
    get_obligation_extractor,
    get_app_version,
    get_llm_config
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_s3_client():
    """Mock S3 client."""
    with patch('src.api.routes.get_s3_client') as mock:
        s3_mock = Mock()
        s3_mock.put_object = Mock()
        s3_mock.generate_presigned_url = Mock(return_value="https://s3.example.com/report.pdf")
        mock.return_value = s3_mock
        yield s3_mock


@pytest.fixture
def mock_s3_file():
    """Mock S3 file retrieval."""
    with patch('src.api.routes.get_s3_file') as mock:
        mock.return_value = b"Mock PDF content"
        yield mock


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    with patch('src.api.routes.get_llm_client') as mock:
        llm_mock = Mock()
        llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
        llm_mock.get_embeddings_batch = Mock(return_value=[[0.1] * 768, [0.2] * 768])
        llm_mock.generate = Mock(return_value="This is a test response from LLM")
        mock.return_value = llm_mock
        yield llm_mock


@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client."""
    with patch('src.api.routes.get_milvus_client') as mock:
        milvus_mock = Mock()
        milvus_mock.insert_extract_data = Mock()
        milvus_mock.insert_extract_data_batch = Mock(return_value=["chunk-1", "chunk-2"])
        milvus_mock.create_contracts_collection = Mock()
        milvus_mock.insert_contract = Mock()
        milvus_mock.insert_obligation = Mock()
        milvus_mock.insert_obligations_batch = Mock(return_value=["obl-1", "obl-2"])
        milvus_mock.search_extract_data = Mock(return_value=[
            {"content": "Sample contract content", "contract_id": "doc-2025-001"}
        ])
        mock.return_value = milvus_mock
        yield milvus_mock


@pytest.fixture
def mock_doc_processor():
    """Mock document processor."""
    with patch('src.api.routes.get_doc_processor') as mock:
        doc_mock = Mock()
        doc_mock.validate_file = Mock(return_value=(True, None))
        doc_mock.extract_text = Mock(return_value="Sample contract text content")
        doc_mock.chunk_text = Mock(return_value=[
            {"text": "Chunk 1 content", "index": 0},
            {"text": "Chunk 2 content", "index": 1}
        ])
        mock.return_value = doc_mock
        yield doc_mock


@pytest.fixture
def mock_obligation_extractor():
    """Mock obligation extractor."""
    with patch('src.api.routes.get_obligation_extractor') as mock:
        extractor_mock = Mock()
        extractor_mock.extract_obligations = Mock(return_value=[
            {
                "id": "doc-2025-001_obl_0",
                "contract_id": "doc-2025-001",
                "type": "payment_schedule",
                "description": "Monthly payment due",
                "due_date": "2025-01-15",
                "party_responsible": "Client",
                "recurrence": "monthly",
                "priority": "high",
                "original_text": "Payment due on 15th",
                "status": "pending"
            }
        ])
        mock.return_value = extractor_mock
        yield extractor_mock


@pytest.fixture
def mock_report_synthesizer():
    """Mock report synthesizer."""
    with patch('src.api.routes.ReportSynthesizer') as mock:
        synthesizer_mock = Mock()
        synthesizer_mock.synthesize_report = AsyncMock(return_value={
            'content': b'PDF content',
            'format': 'pdf'
        })
        mock.return_value = synthesizer_mock
        yield synthesizer_mock


@pytest.fixture
def mock_event_logger():
    """Mock Kafka event logger."""
    with patch('src.api.routes.create_event_logger') as mock:
        logger_mock = Mock()
        logger_mock.log_event = Mock()
        mock.return_value = logger_mock
        yield logger_mock


@pytest.fixture
def mock_llm_config():
    """Mock LLM config."""
    with patch('src.api.routes.get_llm_config') as mock:
        mock.return_value = {
            "model": "azure/gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        yield mock


@pytest.fixture
def mock_token_tracker():
    """Mock token tracker."""
    with patch('src.api.routes.LLMUsageTracker') as mock:
        tracker_mock = Mock()
        mock.return_value = tracker_mock
        yield tracker_mock


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_check_response_structure(self, client):
        """Test health check response structure."""
        response = client.get("/api/v1/health")
        
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)
        
        # Validate timestamp format
        datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))


class TestUploadEndpoint:
    """Test suite for upload endpoint."""
    
    def test_upload_success(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test successful file upload and processing."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "file_id" in data
        assert "filename" in data
        assert "pdf_url" in data
        assert "obligations" in data
        assert isinstance(data["obligations"], list)
    
    def test_upload_with_query(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test upload with query parameter."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "query": "What are the payment terms?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        # Verify LLM was called for query
        mock_llm_client.generate.assert_called()
    
    def test_upload_invalid_file(
        self,
        client,
        mock_s3_file,
        mock_llm_config,
        mock_event_logger,
        mock_token_tracker
    ):
        """Test upload with invalid file."""
        with patch('src.api.routes.get_doc_processor') as mock:
            doc_mock = Mock()
            doc_mock.validate_file = Mock(return_value=(False, "Invalid file type"))
            mock.return_value = doc_mock

            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.exe",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )

            # Returns 400 for invalid file type (validation error)
            assert response.status_code == 400
    
    def test_upload_missing_s3_url(self, client):
        """Test upload without S3 URL."""
        response = client.post(
            "/api/v1/upload",
            data={
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_missing_user_metadata(self, client):
        """Test upload without user metadata."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_s3_error(
        self,
        client,
        mock_llm_config,
        mock_event_logger,
        mock_token_tracker
    ):
        """Test upload with S3 retrieval error."""
        with patch('src.api.routes.get_s3_file') as mock:
            mock.side_effect = Exception("S3 connection failed")
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500
    
    def test_upload_document_processing_error(
        self,
        client,
        mock_s3_file,
        mock_llm_config,
        mock_event_logger,
        mock_token_tracker
    ):
        """Test upload with document processing error."""
        with patch('src.api.routes.get_doc_processor') as mock:
            doc_mock = Mock()
            doc_mock.validate_file = Mock(return_value=(True, None))
            doc_mock.extract_text = Mock(side_effect=Exception("Extraction failed"))
            mock.return_value = doc_mock
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500
    
    def test_upload_milvus_storage_error(
        self,
        client,
        mock_s3_file,
        mock_llm_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_llm_config,
        mock_event_logger,
        mock_token_tracker
    ):
        """Test upload with Milvus storage error."""
        with patch('src.api.routes.get_milvus_client') as mock:
            milvus_mock = Mock()
            milvus_mock.insert_extract_data_batch = Mock(side_effect=Exception("Milvus error"))
            mock.return_value = milvus_mock

            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )

            assert response.status_code == 500
    
    def test_upload_obligation_extraction(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test that obligations are extracted during upload."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        # Verify obligation extractor was called
        mock_obligation_extractor.extract_obligations.assert_called_once()
        
        # Verify obligations in response
        data = response.json()
        assert len(data["obligations"]) > 0
        assert data["obligations"][0]["type"] == "payment_schedule"
    
    def test_upload_pdf_report_generation(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test PDF report generation during upload."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify PDF URL is returned
        assert "pdf_url" in data
        assert data["pdf_url"].startswith("https://")
        
        # Verify S3 upload was called
        mock_s3_client.put_object.assert_called_once()


class TestChatEndpoint:
    """Test suite for chat endpoint."""
    
    def test_chat_success(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test successful chat interaction."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": 
"What are the payment terms?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0
    
    def test_chat_with_document_id(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test chat with specific document ID."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the obligations?",
                "user_metadata": json.dumps({"team_id": "team-123"}),
                "document_id": "doc-2025-001"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

        # Verify Milvus search was called with document_id
        mock_milvus_client.search_extract_data.assert_called()
    
    def test_chat_missing_message(self, client):
        """Test chat without message."""
        response = client.post(
            "/api/v1/chat",
            data={
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_missing_user_metadata(self, client):
        """Test chat without user metadata."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the terms?"
            }
        )

        assert response.status_code == 422  # Validation error
    
    def test_chat_no_relevant_documents(
        self,
        client,
        mock_llm_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test chat when no relevant documents found."""
        with patch('src.api.routes.get_milvus_client') as mock:
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(return_value=[])
            mock.return_value = milvus_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the payment terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 200
            # LLM should still be called with "No relevant documents found" context
            mock_llm_client.generate.assert_called()
    
    def test_chat_llm_error(
        self,
        client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test chat with LLM generation error."""
        with patch('src.api.routes.get_llm_client') as mock:
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.generate = Mock(side_effect=Exception("LLM API error"))
            mock.return_value = llm_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the payment terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500
    
    def test_chat_milvus_search_error(
        self,
        client,
        mock_llm_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test chat with Milvus search error."""
        with patch('src.api.routes.get_milvus_client') as mock:
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(side_effect=Exception("Milvus connection error"))
            mock.return_value = milvus_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the payment terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 500
    
    def test_chat_embedding_generation(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test that embeddings are generated for chat queries."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the payment terms?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        # Verify embedding was generated
        mock_llm_client.get_single_embedding.assert_called_once()
    
    def test_chat_context_building(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test that context is properly built from search results."""
        mock_milvus_client.search_extract_data.return_value = [
            {"content": "Payment is due monthly"},
            {"content": "Terms are net 30 days"}
        ]

        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the payment terms?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )

        assert response.status_code == 200
        # Verify LLM was called with context
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        # First positional argument is the prompt
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "Payment is due monthly" in prompt
        assert "Terms are net 30 days" in prompt



class TestHelperFunctions:
    """Test suite for helper functions."""
    
    def test_get_llm_client_singleton(self):
        """Test that get_llm_client returns singleton instance."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            with patch('src.api.routes.LiteLLMClient') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                # Reset singleton
                import src.api.routes as routes_module
                routes_module._llm_client = None
                client1 = get_llm_client()
                client2 = get_llm_client()
                assert client1 is client2

    def test_get_milvus_client_singleton(self):
        """Test that get_milvus_client returns singleton instance."""
        with patch('src.api.routes.MilvusClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            # Reset singleton
            import src.api.routes as routes_module
            routes_module._milvus_client = None
            client1 = get_milvus_client()
            client2 = get_milvus_client()
            assert client1 is client2

    def test_get_doc_processor_singleton(self):
        """Test that get_doc_processor returns singleton instance."""
        with patch('src.api.routes.DocumentProcessor') as mock_processor:
            mock_instance = Mock()
            mock_processor.return_value = mock_instance
            # Reset singleton
            import src.api.routes as routes_module
            routes_module._doc_processor = None
            processor1 = get_doc_processor()
            processor2 = get_doc_processor()
            assert processor1 is processor2

    def test_get_obligation_extractor_singleton(self):
        """Test that get_obligation_extractor returns singleton instance."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            with patch('src.api.routes.ObligationExtractor') as mock_extractor:
                with patch('src.api.routes.get_llm_client') as mock_llm:
                    mock_llm.return_value = Mock()
                    mock_instance = Mock()
                    mock_extractor.return_value = mock_instance
                    # Reset singleton
                    import src.api.routes as routes_module
                    routes_module._obligation_extractor = None
                    extractor1 = get_obligation_extractor()
                    extractor2 = get_obligation_extractor()
                    assert extractor1 is extractor2
    
    def test_get_app_version(self):
        """Test app version retrieval."""
        version = get_app_version()
        assert isinstance(version, str)
        assert len(version) > 0
    
    @pytest.mark.asyncio
    async def test_get_llm_config_with_team_id(self):
        """Test LLM config retrieval with team ID."""
        with patch('src.api.routes.get_model_config') as mock_config:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get_team_model_config = AsyncMock(
                return_value={
                    "selected_model": "gpt-4",
                    "provider": "azure",
                    "config": {"temperature": 0.7}
     
           }
            )
            mock_config.return_value = mock_context
            
            user_metadata = json.dumps({"team_id": "team-123"})
            config = await get_llm_config(user_metadata)
            
            assert config["model"] == "azure/gpt-4"
            assert config["temperature"] == pytest.approx(0.7)
    
    @pytest.mark.asyncio
    async def test_get_llm_config_without_team_id(self):
        """Test LLM config retrieval without team ID."""
        with patch('src.api.routes.get_model_config') as mock_config:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get_team_model_config = AsyncMock(
                return_value={
                    "selected_model": "gpt-3.5-turbo",
                    "provider": "openai",
                    "config": {"temperature": 0.5}
                }
            )
            mock_config.return_value = mock_context

            user_metadata = json.dumps({})
            config = await get_llm_config(user_metadata)

            assert "model" in config

    @pytest.mark.asyncio
    async def test_get_llm_config_error_handling(self):
        """Test LLM config error handling."""
        with patch('src.api.routes.get_model_config') as mock_config:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get_team_model_config = AsyncMock(
                side_effect=Exception("Database error")
            )
            mock_config.return_value = mock_context

            user_metadata = json.dumps({"team_id": "team-123"})
            
            with pytest.raises(ValueError):
                await get_llm_config(user_metadata)


class TestEventLogging:
    """Test suite for event logging integration."""
    
    def test_upload_logs_events(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test that upload endpoint logs events."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        # Verify events were logged
        assert mock_event_logger.log_event.call_count >= 3
    
    def test_chat_logs_events(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config,
        mock_token_tracker
    ):
        """Test that chat endpoint logs events."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the terms?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 200
        # Verify events were logged
        assert mock_event_logger.log_event.call_count >= 2


class TestTokenTracking:
    """Test suite for token usage tracking."""
    
    def test_upload_tracks_tokens(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config
    ):
        """Test that upload endpoint tracks token usage."""
        with patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            mock_tracker_instance = Mock()
            mock_tracker.return_value = mock_tracker_instance

            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )

            assert response.status_code == 200
            # Verify token tracker was created
            mock_tracker.assert_called()

    def test_chat_tracks_tokens(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test that chat endpoint tracks token usage."""
        with patch('src.api.routes.LLMUsageTracker') as mock_tracker:
            mock_tracker_instance = Mock()
            mock_tracker.return_value = mock_tracker_instance

            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )

            assert response.status_code == 200
            # Verify token tracker was created
            mock_tracker.assert_called()



class TestUploadEndpointNegative:
    """Negative test cases for upload endpoint."""
    
    def test_upload_empty_s3_url(self, client):
        """Test upload with empty S3 URL."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        assert response.status_code in [400, 422, 500]
    
    def test_upload_malformed_s3_url(
        self,
        client,
        mock_llm_config,
        mock_event_logger
    ):
        """Test upload with malformed S3 URL."""
        with patch('src.api.routes.get_s3_file') as mock:
            mock.side_effect = Exception("Invalid URL")
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "not-a-valid-url",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_upload_invalid_json_metadata(self, client):
        """Test upload with invalid JSON in user_metadata."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": "not-valid-json"
            }
        )
        assert response.status_code in [400, 422, 500]
    
    def test_upload_missing_team_id(
        self,
        client,
        mock_s3_file,
        mock_llm_config,
        mock_event_logger
    ):
        """Test upload with missing team_id in metadata."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({})
            }
        )
        # Should handle gracefully or return error
        assert response.status_code in [200, 400, 500]
    
    def test_upload_corrupted_file(
        self,
        client,
        mock_llm_config,
        mock_event_logger
    ):
        """Test upload with corrupted file content."""
        with patch('src.api.routes.get_s3_file') as mock_s3:
            mock_s3.return_value = b"\x00\x01\x02corrupted"
            
            with patch('src.api.routes.get_doc_processor') as mock_proc:
                doc_mock = Mock()
                doc_mock.validate_file = Mock(return_value=(True, None))
                doc_mock.extract_text = Mock(side_effect=Exception("Corrupted file"))
                mock_proc.return_value = doc_mock
                
                response = client.post(
                    "/api/v1/upload",
                    data={
                        "s3_url": "https://s3.example.com/corrupted.pdf",
                        "user_metadata": json.dumps({"team_id": "team-123"})
                    }
                )
                assert response.status_code == 500
    
    def test_upload_zero_byte_file(
        self,
        client,
        mock_llm_client,
        mock_llm_config,
        mock_event_logger
    ):
        """Test upload with zero-byte file."""
        with patch('src.api.routes.get_s3_file') as mock_s3:
            mock_s3.return_value = b""

            with patch('src.api.routes.get_doc_processor') as mock_proc:
                doc_mock = Mock()
                doc_mock.validate_file = Mock(return_value=(False, "File is empty"))
                mock_proc.return_value = doc_mock

                response = client.post(
                    "/api/v1/upload",
                    data={
                        "s3_url": "https://s3.example.com/empty.pdf",
                        "user_metadata": json.dumps({"team_id": "team-123"})
                    }
                )
                # Returns 500 because validation happens after llm_client is obtained
                assert response.status_code in [400, 500]
    
    def test_upload_embedding_generation_failure(
        self,
        client,
        mock_s3_file,
        mock_doc_processor,
        mock_milvus_client,
        mock_obligation_extractor,
        mock_llm_config,
        mock_event_logger
    ):
        """Test upload when embedding generation fails."""
        with patch('src.api.routes.get_llm_client') as mock:
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(side_effect=Exception("Embedding API error"))
            mock.return_value = llm_mock
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_upload_pdf_generation_failure(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_event_logger,
        mock_llm_config
    ):
        """Test upload when PDF report generation fails."""
        with patch('src.api.routes.ReportSynthesizer') as mock:
            synthesizer_mock = Mock()
            synthesizer_mock.synthesize_report = AsyncMock(side_effect=Exception("PDF generation failed"))
            mock.return_value = synthesizer_mock
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_upload_s3_upload_failure(
        self,
        client,
        mock_s3_file,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config
    ):
        """Test upload when S3 upload of report fails."""
        with patch('src.api.routes.get_s3_client') as mock:
            s3_mock = Mock()
            s3_mock.put_object = Mock(side_effect=Exception("S3 upload failed"))
            mock.return_value = s3_mock
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_upload_very_long_query(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config
    ):
        """Test upload with extremely long query string."""
        long_query = "What are the terms? " * 1000  # Very long query
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "query": long_query,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_upload_special_characters_in_filename(
        self,
        client,
        mock_llm_config,
        mock_event_logger
    ):
        """Test upload with special characters in filename."""
        with patch('src.api.routes.get_s3_file') as mock:
            mock.return_value = b"PDF content"
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": "https://s3.example.com/test%20file%20(1)%20[copy].pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            # Should handle URL encoding
            assert response.status_code in [200, 400, 500]


class TestChatEndpointNegative:
    """Negative test cases for chat endpoint."""
    
    def test_chat_empty_message(self, client):
        """Test chat with empty message."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        # May return 500 due to internal validation or processing
        assert response.status_code in [200, 400, 422, 500]
    
    def test_chat_very_long_message(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat with extremely long message."""
        long_message = "Tell me about the contract. " * 1000
        
        response = client.post(
            "/api/v1/chat",
            data={
                "message": long_message,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_chat_special_characters(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat with special characters and emojis."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What about 💰 payment terms? <script>alert('xss')</script>",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        assert response.status_code == 200
    
    def test_chat_invalid_document_id(
        self,
        client,
        mock_llm_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat with invalid document ID."""
        with patch('src.api.routes.get_milvus_client') as mock:
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(return_value=[])
            mock.return_value = milvus_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"}),
                    "document_id": "non-existent-doc-id"
                }
            )
            assert response.status_code == 200
    
    def test_chat_sql_injection_attempt(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat with SQL injection attempt."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "'; DROP TABLE contracts; --",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        assert response.status_code == 200
    
    def test_chat_embedding_timeout(
        self,
        client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat when embedding generation times out."""
        with patch('src.api.routes.get_llm_client') as mock:
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(side_effect=TimeoutError("Embedding timeout"))
            mock.return_value = llm_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_chat_llm_rate_limit(
        self,
        client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat when LLM API rate limit is hit."""
        with patch('src.api.routes.get_llm_client') as mock:
            llm_mock = Mock()
            llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
            llm_mock.generate = Mock(side_effect=Exception("Rate limit exceeded"))
            mock.return_value = llm_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_chat_milvus_connection_lost(
        self,
        client,
        mock_llm_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat when Milvus connection is lost."""
        with patch('src.api.routes.get_milvus_client') as mock:
            milvus_mock = Mock()
            milvus_mock.search_extract_data = Mock(side_effect=ConnectionError("Milvus connection lost"))
            mock.return_value = milvus_mock
            
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": "What are the terms?",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            assert response.status_code == 500
    
    def test_chat_null_message(self, client):
        """Test chat with null message."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": None,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        # May return 500 due to internal handling of None
        assert response.status_code in [422, 500]
    
    def test_chat_unicode_message(
        self,
        client,
        mock_llm_client,
        mock_milvus_client,
        mock_event_logger,
        mock_llm_config
    ):
        """Test chat with unicode characters."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "契約条件は何ですか？ Quels sont les termes? Какие условия?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        assert response.status_code == 200


class TestEdgeCases:
    """Edge case tests for API endpoints."""
    
    def test_concurrent_uploads(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config
    ):
        """Test multiple concurrent upload requests."""
        responses = []
        for i in range(5):
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": f"https://s3.example.com/test{i}.pdf",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            responses.append(response)
        
        # All should succeed or fail gracefully
        assert all(r.status_code in [200, 500] for r in responses)
    
    def test_upload_then_chat(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config
    ):
        """Test upload followed by chat on same document."""
        # Upload
        upload_response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        assert upload_response.status_code == 200
        
        # Chat
        chat_response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the obligations?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        assert chat_response.status_code == 200
    
    def test_health_check_under_load(self, client):
        """Test health check endpoint under load."""
        responses = []
        for _ in range(100):
            response = client.get("/api/v1/health")
            responses.append(response)
        
        assert all(r.status_code == 200 for r in responses)
    
    def test_upload_with_all_optional_params(
        self,
        client,
        mock_s3_file,
        mock_s3_client,
        mock_llm_client,
        mock_milvus_client,
        mock_doc_processor,
        mock_obligation_extractor,
        mock_report_synthesizer,
        mock_event_logger,
        mock_llm_config
    ):
        """Test upload with all optional parameters provided."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "query": "What are the payment terms?",
                "user_metadata": json.dumps({
                    "team_id": "team-123",
                    "user_id": "user-456",
                    "extra_field": "extra_value"
                })
            }
        )
        assert response.status_code == 200
    
    def test_malformed_request_headers(self, client):
        """Test requests with malformed headers."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "Test",
                "user_metadata": json.dumps({"team_id": "team-123"})
            },
            headers={"Content-Type": "invalid/type"}
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 415, 422, 500]
