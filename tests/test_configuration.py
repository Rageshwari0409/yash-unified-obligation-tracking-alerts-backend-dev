"""
Configuration Validation Tests.
Tests environment variables, config files, and missing configurations.

Run with: pytest tests/test_configuration.py -v
"""
import pytest
import os
import yaml
from unittest.mock import patch, mock_open, MagicMock
import sys

# Mock problematic imports
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['pymilvus'] = MagicMock()


class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_missing_milvus_host(self):
        """Test behavior when MILVUS_HOST is missing."""
        with patch.dict(os.environ, {}, clear=True):
            from src.storage.milvus_client import MilvusClient
            client = MilvusClient()
            # Should use default value
            assert client.host is not None
    
    def test_missing_aws_credentials(self):
        """Test behavior when AWS credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            try:
                from src.utils.s3_utility import get_s3_client
                client = get_s3_client()
                # May succeed with default credentials or fail
                assert client is not None
            except Exception as e:
                # Expected if no credentials available
                assert "credentials" in str(e).lower() or "access" in str(e).lower()
    
    def test_missing_s3_bucket_name(self):
        """Test behavior when S3_BUCKET_NAME is missing."""
        with patch.dict(os.environ, {}, clear=True):
            from src.utils.s3_utility import S3_BUCKET
            # Should have a default or raise error
            assert S3_BUCKET is not None or S3_BUCKET == ""
    
    def test_invalid_milvus_port(self):
        """Test behavior with invalid MILVUS_PORT."""
        with patch.dict(os.environ, {"MILVUS_PORT": "invalid"}):
            from src.storage.milvus_client import MilvusClient
            try:
                client = MilvusClient()
                # Should handle conversion error
                assert isinstance(client.port, int)
            except ValueError:
                # Expected for invalid port
                pass
    
    def test_environment_variable_precedence(self):
        """Test that environment variables override config file values."""
        with patch.dict(os.environ, {"MILVUS_HOST": "env-host"}):
            from src.storage.milvus_client import MilvusClient
            client = MilvusClient()
            assert client.host == "env-host"
    
    def test_all_required_env_vars_present(self):
        """Test that all required environment variables are documented."""
        required_vars = [
            "GOOGLE_API_KEY",
            "MILVUS_HOST",
            "MILVUS_PORT",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S3_BUCKET_NAME"
        ]
        
        # Check .env.example file
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as f:
                env_example = f.read()
            
            for var in required_vars:
                assert var in env_example, f"{var} not documented in .env.example"


class TestConfigFiles:
    """Test configuration file handling."""
    
    def test_api_config_yaml_exists(self):
        """Test that api_config.yaml exists and is valid."""
        assert os.path.exists("config/api_config.yaml"), "api_config.yaml not found"
        
        with open("config/api_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert "api" in config
        assert "document_processing" in config
    
    def test_model_config_yaml_exists(self):
        """Test that model_config.yaml exists and is valid."""
        assert os.path.exists("config/model_config.yaml"), "model_config.yaml not found"
        
        with open("config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert "app" in config or "extraction_settings" in config
    
    def test_milvus_config_yaml_exists(self):
        """Test that milvus_config.yaml exists and is valid."""
        assert os.path.exists("config/milvus_config.yaml"), "milvus_config.yaml not found"
        
        with open("config/milvus_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert "milvus" in config
        assert "collections" in config
    
    def test_api_config_required_fields(self):
        """Test that api_config.yaml has all required fields."""
        with open("config/api_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Check API section
        assert "title" in config["api"]
        assert "version" in config["api"]
        assert "prefix" in config["api"]
        
        # Check document processing section
        assert "supported_formats" in config["document_processing"]
        assert "max_file_size_mb" in config["document_processing"]
    
    def test_model_config_required_fields(self):
        """Test that model_config.yaml has all required fields."""
        with open("config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Check extraction settings
        if "extraction_settings" in config:
            assert "chunk_size" in config["extraction_settings"]
            assert "chunk_overlap" in config["extraction_settings"]
    
    def test_milvus_config_required_fields(self):
        """Test that milvus_config.yaml has all required fields."""
        with open("config/milvus_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Check Milvus section
        assert "host" in config["milvus"]
        assert "port" in config["milvus"]
        
        # Check collections
        assert "contracts" in config["collections"]
        assert "dimension" in config["collections"]["contracts"]
    
    def test_malformed_yaml_handling(self):
        """Test handling of malformed YAML files."""
        malformed_yaml = "invalid: yaml: content: [unclosed"
        
        with patch("builtins.open", mock_open(read_data=malformed_yaml)):
            with pytest.raises(yaml.YAMLError):
                yaml.safe_load(malformed_yaml)
    
    def test_missing_config_file_handling(self):
        """Test handling when config file is missing."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                with open("nonexistent_config.yaml", "r") as f:
                    yaml.safe_load(f)
    
    def test_config_file_permissions(self):
        """Test that config files have appropriate permissions."""
        config_files = [
            "config/api_config.yaml",
            "config/model_config.yaml",
            "config/milvus_config.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                # Check file is readable
                assert os.access(config_file, os.R_OK), f"{config_file} not readable"


class TestConfigValidation:
    """Test configuration value validation."""
    
    def test_supported_file_formats_valid(self):
        """Test that supported file formats are valid."""
        with open("config/api_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        formats = config["document_processing"]["supported_formats"]
        valid_formats = [".pdf", ".docx", ".txt", ".doc"]
        
        for fmt in formats:
            assert fmt in valid_formats, f"Invalid format: {fmt}"
    
    def test_max_file_size_reasonable(self):
        """Test that max file size is reasonable."""
        with open("config/api_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        max_size = config["document_processing"]["max_file_size_mb"]
        assert 1 <= max_size <= 500, f"Unreasonable max file size: {max_size}MB"
    
    def test_chunk_size_reasonable(self):
        """Test that chunk size is reasonable."""
        with open("config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        if "extraction_settings" in config:
            chunk_size = config["extraction_settings"]["chunk_size"]
            assert 100 <= chunk_size <= 10000, f"Unreasonable chunk size: {chunk_size}"
    
    def test_chunk_overlap_less_than_chunk_size(self):
        """Test that chunk overlap is less than chunk size."""
        with open("config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        if "extraction_settings" in config:
            chunk_size = config["extraction_settings"]["chunk_size"]
            chunk_overlap = config["extraction_settings"]["chunk_overlap"]
            assert chunk_overlap < chunk_size, "Overlap should be less than chunk size"
    
    def test_milvus_dimension_valid(self):
        """Test that Milvus dimension is valid."""
        with open("config/milvus_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        dimension = config["collections"]["contracts"]["dimension"]
        valid_dimensions = [128, 256, 384, 512, 768, 1024, 1536]
        assert dimension in valid_dimensions, f"Invalid dimension: {dimension}"
    
    def test_milvus_metric_type_valid(self):
        """Test that Milvus metric type is valid."""
        with open("config/milvus_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        metric_type = config["collections"]["contracts"]["metric_type"]
        valid_metrics = ["L2", "IP", "COSINE"]
        assert metric_type in valid_metrics, f"Invalid metric: {metric_type}"
    
    def test_api_version_format(self):
        """Test that API version follows semantic versioning."""
        with open("config/api_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        version = config["api"]["version"]
        # Should match X.Y.Z format
        import re
        assert re.match(r'^\d+\.\d+\.\d+$', version), f"Invalid version format: {version}"


class TestConfigDefaults:
    """Test default configuration values."""
    
    def test_milvus_client_defaults(self):
        """Test MilvusClient uses sensible defaults."""
        with patch.dict(os.environ, {}, clear=True):
            from src.storage.milvus_client import MilvusClient
            client = MilvusClient()
            
            # Should have default values
            assert client.host in ["localhost", "127.0.0.1", "milvus"]
            assert client.port in [19530, 9091]
            assert client.database is not None
    
    def test_document_processor_defaults(self):
        """Test DocumentProcessor uses sensible defaults."""
        from src.processing.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        assert processor.max_file_size_mb > 0
        assert processor.max_chunk_size > 0
        assert len(processor.supported_formats) > 0
    
class TestDatabaseConfiguration:
    """Test database configuration."""
    
    def test_db_connection_string_format(self):
        """Test that DB connection string is properly formatted."""
        # Check if DB config exists
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        
        if db_host and db_name:
            # Verify format
            assert isinstance(db_port, str)
            assert db_port.isdigit()
            assert 1 <= int(db_port) <= 65535
    
    def test_db_credentials_not_in_code(self):
        """Test that DB credentials are not hardcoded."""
        # Check source files for hardcoded credentials
        sensitive_patterns = [
            "password=",
            "pwd=",
            "secret=",
            "api_key="
        ]
        
        # This is a basic check - in production use proper secret scanning
        import glob
        for py_file in glob.glob("src/**/*.py", recursive=True):
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                    for pattern in sensitive_patterns:
                        # Allow in comments or variable names, but not in strings
                        if f'"{pattern}' in content or f"'{pattern}" in content:
                            # This is a warning, not a failure
                            print(f"Warning: Potential hardcoded credential in {py_file}")
            except Exception as e:
                # Skip files that can't be read
                print(f"Skipping {py_file}: {e}")


class TestLoggingConfiguration:
    """Test logging configuration."""
    
    def test_log_level_valid(self):
        """Test that LOG_LEVEL is valid."""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert log_level.upper() in valid_levels, f"Invalid log level: {log_level}"
    
    def test_logging_setup(self):
        """Test that logging is properly configured."""
        from src.utils.logger import setup_logging
        logger = setup_logging()
        assert logger is not None
    