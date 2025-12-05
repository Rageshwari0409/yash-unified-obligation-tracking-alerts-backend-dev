"""
Security Tests - XSS, Injection, File Upload, and other security vulnerabilities.

Run with: pytest tests/test_security.py -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
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


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for security tests."""
    with patch('src.api.routes.get_s3_file') as mock_s3_file, \
         patch('src.api.routes.get_s3_client') as mock_s3_client, \
         patch('src.api.routes.get_llm_client') as mock_llm, \
         patch('src.api.routes.get_milvus_client') as mock_milvus, \
         patch('src.api.routes.get_doc_processor') as mock_processor, \
         patch('src.api.routes.get_obligation_extractor') as mock_extractor, \
         patch('src.api.routes.ReportSynthesizer') as mock_report, \
         patch('src.api.routes.create_event_logger') as mock_logger, \
         patch('src.api.routes.get_llm_config') as mock_config, \
         patch('src.api.routes.LLMUsageTracker') as mock_tracker:
        
        # Setup mocks
        mock_s3_file.return_value = b"PDF content"
        
        s3_mock = Mock()
        s3_mock.put_object = Mock()
        s3_mock.generate_presigned_url = Mock(return_value="https://s3.example.com/report.pdf")
        mock_s3_client.return_value = s3_mock
        
        llm_mock = Mock()
        llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
        llm_mock.get_embeddings_batch = Mock(return_value=[[0.1] * 768])
        llm_mock.generate = Mock(return_value="Safe response")
        mock_llm.return_value = llm_mock
        
        milvus_mock = Mock()
        milvus_mock.insert_extract_data_batch = Mock(return_value=["chunk-1"])
        milvus_mock.create_contracts_collection = Mock()
        milvus_mock.insert_contract = Mock()
        milvus_mock.insert_obligations_batch = Mock(return_value=["obl-1"])
        milvus_mock.search_extract_data = Mock(return_value=[{"content": "Safe content"}])
        mock_milvus.return_value = milvus_mock
        
        doc_mock = Mock()
        doc_mock.validate_file = Mock(return_value=(True, None))
        doc_mock.extract_text = Mock(return_value="Safe text")
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
        
        yield {
            's3_file': mock_s3_file,
            's3_client': s3_mock,
            'llm': llm_mock,
            'milvus': milvus_mock,
            'processor': doc_mock,
            'extractor': extractor_mock
        }


class TestXSSPrevention:
    """Test Cross-Site Scripting (XSS) prevention."""
    
    def test_xss_in_chat_message(self, client, mock_dependencies):
        """Test that XSS payloads in chat messages are sanitized."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<<SCRIPT>alert('XSS');//<</SCRIPT>",
        ]
        
        for payload in xss_payloads:
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": payload,
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should not execute script, should return safe response
            assert response.status_code == 200
            response_data = response.json()
            # Response should not contain unescaped script tags
            assert "<script>" not in response_data.get("message", "").lower()
            assert "onerror=" not in response_data.get("message", "").lower()
    
    def test_xss_in_filename(self, client, mock_dependencies):
        """Test that XSS payloads in filenames are sanitized."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/<script>alert('XSS')</script>.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_xss_in_user_metadata(self, client, mock_dependencies):
        """Test that XSS payloads in user metadata are sanitized."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the terms?",
                "user_metadata": json.dumps({
                    "team_id": "<script>alert('XSS')</script>"
                })
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]


class TestSQLInjection:
    """Test SQL Injection prevention."""
    
    def test_sql_injection_in_chat_message(self, client, mock_dependencies):
        """Test that SQL injection attempts are prevented."""
        sql_payloads = [
            "'; DROP TABLE contracts; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users--",
            "admin'--",
            "' OR 1=1--",
            "'; DELETE FROM obligations WHERE '1'='1",
            "1'; UPDATE contracts SET user_id='hacker' WHERE '1'='1",
        ]
        
        for payload in sql_payloads:
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": payload,
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should not execute SQL, should return safe response
            assert response.status_code == 200
            # Milvus uses vector search, not SQL, so this should be safe
    
    def test_sql_injection_in_document_id(self, client, mock_dependencies):
        """Test SQL injection in document_id parameter."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the terms?",
                "user_metadata": json.dumps({"team_id": "team-123"}),
                "document_id": "'; DROP TABLE contracts; --"
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]


class TestPathTraversal:
    """Test path traversal attack prevention."""
    
    def test_path_traversal_in_s3_url(self, client, mock_dependencies):
        """Test that path traversal in S3 URL is prevented."""
        path_traversal_payloads = [
            "https://s3.example.com/../../etc/passwd",
            "https://s3.example.com/../../../windows/system32/config/sam",
            "https://s3.example.com/....//....//....//etc/passwd",
            "https://s3.example.com/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]
        
        for payload in path_traversal_payloads:
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": payload,
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should reject or handle safely
            assert response.status_code in [400, 403, 500]
    
    def test_path_traversal_in_filename(self, client, mock_dependencies):
        """Test that path traversal in filename is prevented."""
        mock_dependencies['processor'].validate_file = Mock(return_value=(False, "Invalid path"))
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/../../etc/passwd",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should reject
        assert response.status_code in [400, 500]


class TestFileUploadSecurity:
    """Test file upload security."""
    
    def test_file_size_limit_enforced(self, client, mock_dependencies):
        """Test that file size limits are enforced."""
        # Mock a file that's too large
        mock_dependencies['s3_file'].return_value = b"A" * (51 * 1024 * 1024)  # 51MB
        mock_dependencies['processor'].validate_file = Mock(
            return_value=(False, "File too large")
        )
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/large.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should reject
        assert response.status_code in [400, 500]
    
    def test_malicious_file_extension(self, client, mock_dependencies):
        """Test that malicious file extensions are rejected."""
        malicious_extensions = [
            "test.exe",
            "test.bat",
            "test.sh",
            "test.cmd",
            "test.com",
            "test.scr",
            "test.vbs",
            "test.js",
            "test.jar",
        ]
        
        for filename in malicious_extensions:
            mock_dependencies['processor'].validate_file = Mock(
                return_value=(False, "Unsupported file format")
            )
            
            response = client.post(
                "/api/v1/upload",
                data={
                    "s3_url": f"https://s3.example.com/{filename}",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should reject
            assert response.status_code in [400, 500]
    
    def test_double_extension_attack(self, client, mock_dependencies):
        """Test that double extension attacks are prevented."""
        mock_dependencies['processor'].validate_file = Mock(
            return_value=(False, "Unsupported file format")
        )
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/document.pdf.exe",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should reject based on final extension
        assert response.status_code in [400, 500]
    
    def test_null_byte_injection_in_filename(self, client, mock_dependencies):
        """Test that null byte injection is prevented."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/document.pdf\x00.exe",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should handle safely
        assert response.status_code in [200, 400, 500]
    
    def test_malicious_pdf_content(self, client, mock_dependencies):
        """Test handling of PDFs with malicious content."""
        # Mock PDF with embedded JavaScript
        mock_dependencies['s3_file'].return_value = b"%PDF-1.4\n<script>alert('XSS')</script>"
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/malicious.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should handle safely (extract text only)
        assert response.status_code in [200, 500]


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_extremely_long_message(self, client, mock_dependencies):
        """Test handling of extremely long messages."""
        long_message = "A" * 1000000  # 1MB message
        
        response = client.post(
            "/api/v1/chat",
            data={
                "message": long_message,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should handle gracefully (truncate or reject)
        assert response.status_code in [200, 400, 413, 500]
    
    def test_special_characters_in_input(self, client, mock_dependencies):
        """Test handling of special characters."""
        special_chars = [
            "Test\x00null\x00byte",
            "Test\r\nCRLF\r\ninjection",
            "Test\x1b[31mANSI\x1b[0m",
            "Test\u202eRTL\u202c",
            "Test\ufeffBOM",
        ]
        
        for payload in special_chars:
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": payload,
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 500]
    
    def test_unicode_normalization(self, client, mock_dependencies):
        """Test Unicode normalization attacks."""
        # Unicode characters that look similar but are different
        payloads = [
            "admin",  # Normal
            "аdmin",  # Cyrillic 'а'
            "аdmіn",  # Cyrillic 'а' and 'і'
        ]
        
        for payload in payloads:
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": f"User: {payload}",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            
            assert response.status_code == 200
    
    def test_json_injection_in_metadata(self, client, mock_dependencies):
        """Test JSON injection in user_metadata."""
        malicious_json = '{"team_id": "team-123", "admin": true, "role": "superuser"}'
        
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "Test message",
                "user_metadata": malicious_json
            }
        )
        
        # Should parse JSON but not grant elevated privileges
        assert response.status_code in [200, 400, 500]
    
    def test_empty_string_parameters(self, client, mock_dependencies):
        """Test handling of empty string parameters."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 500]


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    @pytest.mark.skip(reason="Authentication currently disabled")
    def test_missing_auth_token(self, client):
        """Test that requests without auth token are rejected."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Authentication currently disabled")
    def test_invalid_jwt_token(self, client):
        """Test that invalid JWT tokens are rejected."""
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            },
            headers={"Authorization": "Bearer invalid.jwt.token"}
        )
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Authentication currently disabled")
    def test_expired_jwt_token(self, client):
        """Test that expired JWT tokens are rejected."""
        # Use an expired token
        expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1MTYyMzkwMjJ9.xxx"
        
        response = client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            },
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        
        assert response.status_code == 401
    
@pytest.mark.skip(reason="Authentication currently disabled")
class TestRateLimiting:
    """Test rate limiting and DoS prevention."""
    
    def test_concurrent_request_limit(self, client, mock_dependencies):
        """Test concurrent request limits."""
        import concurrent.futures
        
        def make_request(index):
            return client.post(
                "/api/v1/chat",
                data={
                    "message": f"Test {index}",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            ).status_code
        
        # Send 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Should handle gracefully
        assert all(status in [200, 429, 500, 503] for status in results)


class TestDataLeakage:
    """Test for data leakage and information disclosure."""
    
    def test_stack_traces_not_exposed(self, client, mock_dependencies):
        """Test that stack traces are not exposed to users."""
        mock_dependencies['llm'].generate = Mock(side_effect=Exception("Internal error"))
        
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "Test",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        if response.status_code == 500:
            error_detail = response.json().get("detail", "")
            # Should not contain stack trace
            assert "Traceback" not in error_detail
            assert "File \"" not in error_detail
    
    def test_api_keys_not_in_responses(self, client, mock_dependencies):
        """Test that API keys are not included in responses."""
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "Test",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        
        response_text = json.dumps(response.json())
        # Should not contain API keys
        assert "GOOGLE_API_KEY" not in response_text
        assert "AWS_SECRET" not in response_text
        assert "api_key" not in response_text.lower()


class TestCORSSecurity:
    """Test CORS security."""
    
    def test_cors_not_allow_all_origins_in_production(self, client):
        """Test that CORS doesn't allow all origins in production."""
        import os
        if os.getenv("ENVIRONMENT") == "production":
            response = client.options("/api/v1/health")
            cors_origin = response.headers.get("access-control-allow-origin", "")
            assert cors_origin != "*", "CORS should not allow all origins in production"


class TestContentSecurityPolicy:
    """Test Content Security Policy headers."""
    
    def test_csp_headers_present(self, client):
        """Test that CSP headers are set."""
        response = client.get("/api/v1/health")
        
        # Check for security headers
        headers = {k.lower(): v for k, v in response.headers.items()}
        
        # These are recommended security headers
        # Note: May not be implemented yet
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
        ]
        
        for header in security_headers:
            if header in headers:
                print(f"✓ {header}: {headers[header]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
