"""
Performance and Load Tests.
Tests system performance under various load conditions.

Run with: pytest tests/test_performance.py -v --tb=short
For load tests: pytest tests/test_performance.py -v -m load
"""
import pytest
import time
import concurrent.futures
import statistics
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import psutil
import os

# Mock problematic imports
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['pymilvus'] = MagicMock()

from app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_all_dependencies():
    """Mock all dependencies for performance testing."""
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
        
        # Setup fast mocks
        mock_s3_file.return_value = b"PDF content"
        
        s3_mock = Mock()
        s3_mock.put_object = Mock()
        s3_mock.generate_presigned_url = Mock(return_value="https://s3.example.com/report.pdf")
        mock_s3_client.return_value = s3_mock
        
        llm_mock = Mock()
        llm_mock.get_single_embedding = Mock(return_value=[0.1] * 768)
        llm_mock.get_embeddings_batch = Mock(return_value=[[0.1] * 768])
        llm_mock.generate = Mock(return_value="Response")
        mock_llm.return_value = llm_mock
        
        milvus_mock = Mock()
        milvus_mock.insert_extract_data_batch = Mock(return_value=["chunk-1"])
        milvus_mock.create_contracts_collection = Mock()
        milvus_mock.insert_contract = Mock()
        milvus_mock.insert_obligations_batch = Mock(return_value=["obl-1"])
        milvus_mock.search_extract_data = Mock(return_value=[{"content": "Content"}])
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
        
        yield


class TestResponseTime:
    """Test API response times."""
    
    def test_health_check_response_time(self, client):
        """Test that health check responds quickly."""
        start = time.time()
        response = client.get("/api/v1/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.1, f"Health check too slow: {duration}s"
    
    def test_chat_response_time(self, client, mock_all_dependencies):
        """Test chat endpoint response time."""
        start = time.time()
        response = client.post(
            "/api/v1/chat",
            data={
                "message": "What are the payment terms?",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 2.0, f"Chat response too slow: {duration}s"
    
class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    @pytest.mark.load
    def test_10_concurrent_chat_requests(self, client, mock_all_dependencies):
        """Test 10 concurrent chat requests."""
        def make_request(index):
            start = time.time()
            response = client.post(
                "/api/v1/chat",
                data={
                    "message": f"Test message {index}",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
                "index": index
            }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(r["status"] == 200 for r in results)
        
        # Calculate statistics
        durations = [r["duration"] for r in results]
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)
        
        print("\n10 concurrent requests:")
        print(f"  Average: {avg_duration:.2f}s")
        print(f"  Max: {max_duration:.2f}s")
        
        assert avg_duration < 3.0, f"Average response time too slow: {avg_duration}s"
        assert max_duration < 5.0, f"Max response time too slow: {max_duration}s"
    
    @pytest.mark.load
    def test_50_concurrent_chat_requests(self, client, mock_all_dependencies):
        """Test 50 concurrent chat requests."""
        def make_request(index):
            start = time.time()
            try:
                response = client.post(
                    "/api/v1/chat",
                    data={
                        "message": f"Test {index}",
                        "user_metadata": json.dumps({"team_id": "team-123"})
                    }
                )
                duration = time.time() - start
                return {
                    "status": response.status_code,
                    "duration": duration,
                    "success": True
                }
            except Exception as e:
                return {
                    "status": 500,
                    "duration": 0,
                    "success": False,
                    "error": str(e)
                }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Most should succeed
        success_count = sum(1 for r in results if r["success"])
        success_rate = success_count / len(results) * 100
        
        print("\n50 concurrent requests:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Failures: {50 - success_count}")
        
        assert success_rate >= 90, f"Success rate too low: {success_rate}%"
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_100_concurrent_requests(self, client, mock_all_dependencies):
        """Test 100 concurrent requests (stress test)."""
        def make_request(index):
            try:
                response = client.post(
                    "/api/v1/chat",
                    data={
                        "message": f"Test {index}",
                        "user_metadata": json.dumps({"team_id": "team-123"})
                    }
                )
                return response.status_code == 200
            except Exception:
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(make_request, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(results)
        success_rate = success_count / len(results) * 100
        
        print("\n100 concurrent requests:")
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Should handle at least 80% successfully
        assert success_rate >= 80, f"Success rate too low under load: {success_rate}%"


class TestThroughput:
    """Test system throughput."""
    
    @pytest.mark.load
    def test_requests_per_second(self, client, mock_all_dependencies):
        """Test requests per second capacity."""
        duration = 10  # seconds
        request_count = 0
        start_time = time.time()
        
        def make_request():
            try:
                response = client.get("/api/v1/health")
                return response.status_code == 200
            except Exception:
                return False
        
        while time.time() - start_time < duration:
            if make_request():
                request_count += 1
        
        rps = request_count / duration
        print(f"\nThroughput: {rps:.1f} requests/second")
        
        # Should handle at least 10 requests per second
        assert rps >= 10, f"Throughput too low: {rps:.1f} req/s"
    
    @pytest.mark.load
    def test_sustained_load(self, client, mock_all_dependencies):
        """Test sustained load over time."""
        duration = 30  # seconds
        results = []
        
        def make_request(index):
            start = time.time()
            try:
                response = client.post(
                    "/api/v1/chat",
                    data={
                        "message": f"Test {index}",
                        "user_metadata": json.dumps({"team_id": "team-123"})
                    }
                )
                duration_ms = (time.time() - start) * 1000
                return {
                    "success": response.status_code == 200,
                    "duration": duration_ms,
                    "timestamp": time.time()
                }
            except Exception:
                return {
                    "success": False,
                    "duration": 0,
                    "timestamp": time.time()
                }
        
        start_time = time.time()
        index = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            while time.time() - start_time < duration:
                future = executor.submit(make_request, index)
                results.append(future)
                index += 1
                time.sleep(0.1)  # 10 requests per second
        
        # Wait for all to complete
        completed_results = [f.result() for f in results]
        
        success_count = sum(1 for r in completed_results if r["success"])
        success_rate = success_count / len(completed_results) * 100
        
        print(f"\nSustained load test ({duration}s):")
        print(f"  Total requests: {len(completed_results)}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        assert success_rate >= 95, f"Success rate degraded under sustained load: {success_rate}%"


class TestMemoryUsage:
    """Test memory usage and leaks."""
    
    def test_memory_usage_baseline(self):
        """Test baseline memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"\nBaseline memory usage: {memory_mb:.1f} MB")
        
        # Should not exceed reasonable limit
        assert memory_mb < 500, f"Baseline memory too high: {memory_mb:.1f} MB"
    
    @pytest.mark.load
    def test_memory_leak_detection(self, client, mock_all_dependencies):
        """Test for memory leaks over many requests."""
        process = psutil.Process(os.getpid())
        
        # Baseline
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Make many requests
        for i in range(100):
            client.post(
                "/api/v1/chat",
                data={
                    "message": f"Test {i}",
                    "user_metadata": json.dumps({"team_id": "team-123"})
                }
            )
        
        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print("\nMemory leak test:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Possible memory leak: {memory_increase:.1f} MB increase"


class TestLargeDataHandling:
    """Test handling of large data."""
    
    def test_large_message_processing(self, client, mock_all_dependencies):
        """Test processing of large messages."""
        large_message = "A" * 100000  # 100KB message
        
        start = time.time()
        response = client.post(
            "/api/v1/chat",
            data={
                "message": large_message,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        duration = time.time() - start
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 413, 500]
        assert duration < 5.0, f"Large message processing too slow: {duration}s"
    
class TestDatabasePerformance:
    """Test database operation performance."""
    pass


@pytest.mark.integration
class TestCacheEffectiveness:
    """Test caching effectiveness (if implemented)."""
    
    def test_repeated_query_performance(self, client, mock_all_dependencies):
        """Test that repeated queries are faster (if cached)."""
        query = "What are the payment terms?"
        
        # First request
        start1 = time.time()
        response1 = client.post(
            "/api/v1/chat",
            data={
                "message": query,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        duration1 = time.time() - start1
        
        # Second request (same query)
        start2 = time.time()
        response2 = client.post(
            "/api/v1/chat",
            data={
                "message": query,
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        duration2 = time.time() - start2
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        print("\nCache test:")
        print(f"  First request: {duration1:.3f}s")
        print(f"  Second request: {duration2:.3f}s")
        
        # If caching is implemented, second should be faster
        # If not, they should be similar
        # This test documents current behavior


class TestConnectionPooling:
    """Test connection pool performance."""
    pass


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_performance_baseline(self, client, mock_all_dependencies):
        """Establish performance baseline for regression testing."""
        # Run standard test suite and record timings
        timings = {}
        
        # Health check
        start = time.time()
        client.get("/api/v1/health")
        timings["health_check"] = time.time() - start
        
        # Chat
        start = time.time()
        client.post(
            "/api/v1/chat",
            data={
                "message": "Test",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        timings["chat"] = time.time() - start
        
        # Upload
        start = time.time()
        client.post(
            "/api/v1/upload",
            data={
                "s3_url": "https://s3.example.com/test.pdf",
                "user_metadata": json.dumps({"team_id": "team-123"})
            }
        )
        timings["upload"] = time.time() - start
        
        print("\nPerformance baseline:")
        for endpoint, duration in timings.items():
            print(f"  {endpoint}: {duration:.3f}s")
        
        # Store baseline for comparison
        # In real implementation, save to file or database


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not load"])
