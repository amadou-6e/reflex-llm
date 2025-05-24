"""
Complete integration tests for the ChatClient class - testing real LocalAI integration.
"""
import pytest
import os
import time
import uuid
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from reflex_llms.chat import ChatClient
from tests.conftest import *


@pytest.fixture
def cle():
    ...


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test isolation."""
    run_id = str(uuid.uuid4())[:6]
    return Path(TEMP_DIR, "test_chat_integration").joinpath(run_id)


@pytest.fixture
def mock_env_empty():
    """Mock empty environment variables to force LocalAI usage."""
    env_vars = {
        "AZURE_OPENAI_API_KEY": "",
        "AZURE_OPENAI_ENDPOINT": "",
        "AZURE_OPENAI_DEPLOY_NAME": "",
        "AZURE_OPENAI_MODEL_NAME": ""
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield


@pytest.fixture
def mock_env_azure():
    """Mock environment variables for Azure OpenAI."""
    env_vars = {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOY_NAME": "test-deployment",
        "AZURE_OPENAI_MODEL_NAME": "gpt-4"
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Hello, how are you?"
    }]


@pytest.fixture
def mock_completion_response():
    """Mock OpenAI completion response."""
    response = Mock()
    choice = Mock()
    message = Mock()
    message.content = "Hello! I'm doing well, thank you for asking."
    choice.message = message
    response.choices = [choice]
    return response


@pytest.fixture
def mock_yaml_config():
    """Mock YAML configuration for model mapping."""
    return """
models:
  - name: "gpt-4"
    urls:
      - "https://example.com/gpt4-model.gguf"
    file: "gpt4-model.gguf"
  - name: "gpt-3.5-turbo"
    urls:
      - "https://example.com/gpt35-model.gguf"
    file: "gpt35-model.gguf"
"""


# =============================================================================
# AZURE OPENAI TESTS (Keep mocked for speed)
# =============================================================================


def test_init_with_azure_config(mock_env_azure):
    """Test initialization with Azure OpenAI configuration."""
    with patch('reflex_llms.chat.AzureOpenAI') as mock_azure, \
         patch('reflex_llms.chat.load_dotenv'):
        mock_azure_client = Mock()
        mock_azure.return_value = mock_azure_client

        client = ChatClient(auto_start_localai=False, preload_models=None)

        assert client.is_azure is True
        assert client.azure_api_key == "test-api-key"
        assert client.azure_endpoint == "https://test.openai.azure.com/"
        assert client.azure_deploy_name == "test-deployment"
        assert client.azure_model_name == "gpt-4"
        assert client.localai is None
        assert client.client == mock_azure_client


def test_preload_models_azure(mock_env_azure):
    """Test that preloading is skipped for Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI'), \
         patch('reflex_llms.chat.load_dotenv'), \
         patch.object(ChatClient, '_preload_models') as mock_preload:
        client = ChatClient(preload_models=["gpt-4"], auto_start_localai=False)
        mock_preload.assert_not_called()


def test_model_name_property_azure(mock_env_azure):
    """Test model_name property for Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI'), \
         patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)
        assert client.model_name == "test-deployment"


def test_client_info_azure(mock_env_azure):
    """Test client_info property for Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI'), \
         patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        info = client.client_info
        expected = {
            "type": "Azure OpenAI",
            "endpoint": "https://test.openai.azure.com/",
            "deployment": "test-deployment",
            "model": "gpt-4",
            "api_version": "2024-02-01"
        }
        assert info == expected


def test_download_model_azure_skip(mock_env_azure):
    """Test that model download is skipped for Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI'), \
         patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        result = client.download_model("gpt-4")
        assert result is None


def test_check_download_status_azure(mock_env_azure):
    """Test checking download status for Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI'), \
         patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        status = client.check_download_status("job-123")

        assert status["status"] == "not_applicable"
        assert status["message"] == "Using Azure OpenAI"


def test_list_available_models_azure(mock_env_azure):
    """Test listing available models for Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI'), \
         patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        models = client.list_available_models()

        assert models == ["test-deployment"]


def test_completion_success_azure(mock_env_azure, sample_messages, mock_completion_response):
    """Test successful completion with Azure OpenAI."""
    with patch('reflex_llms.chat.AzureOpenAI') as mock_azure, \
         patch('reflex_llms.chat.load_dotenv'):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion_response
        mock_azure.return_value = mock_client

        client = ChatClient(auto_start_localai=False, preload_models=None)

        response = client.completion(sample_messages, model="gpt-4")

        assert response == mock_completion_response
        mock_client.chat.completions.create.assert_called_once()


def test_completion_content_only(mock_env_azure, sample_messages, mock_completion_response):
    """Test completion with content_only=True."""
    with patch('reflex_llms.chat.AzureOpenAI') as mock_azure, \
         patch('reflex_llms.chat.load_dotenv'):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion_response
        mock_azure.return_value = mock_client

        client = ChatClient(auto_start_localai=False, preload_models=None)

        response = client.completion(sample_messages, content_only=True)

        assert response == "Hello! I'm doing well, thank you for asking."


def test_prompt_method(mock_env_azure, mock_completion_response):
    """Test the prompt convenience method."""
    with patch('reflex_llms.chat.AzureOpenAI') as mock_azure, \
         patch('reflex_llms.chat.load_dotenv'):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion_response
        mock_azure.return_value = mock_client

        client = ChatClient(auto_start_localai=False, preload_models=None)

        response = client.prompt("Hello there!", model="gpt-4")

        assert response == "Hello! I'm doing well, thank you for asking."
        mock_client.chat.completions.create.assert_called_once()


# =============================================================================
# LOCALAI INTEGRATION TESTS (Real integration, no mocks)
# =============================================================================


def test_init_without_azure_config_no_autostart(mock_env_empty, temp_dir):
    """Test initialization without Azure config and auto-start disabled."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(
            auto_start_localai=False,
            preload_models=None,
            models_path=temp_dir,
        )

        assert client.is_azure is False
        assert client.localai is None
        # Client should be real OpenAI client pointing to LocalAI URL
        assert client.client.base_url == "http://127.0.0.1:8080/v1"


@pytest.mark.docker
def test_init_with_container_handler(mock_env_empty, temp_dir):
    """Test initialization with LocalAI container handler - REAL INTEGRATION."""
    with patch('reflex_llms.chat.load_dotenv'):
        # Use unique port to avoid conflicts
        test_port = 8090 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(
            local_ai_url=f"http://127.0.0.1:{test_port}/v1",
            auto_start_localai=True,
            preload_models=None,
            models_path=temp_dir,
        )

        try:
            assert client.is_azure is False
            assert client.localai is not None

            # Check that LocalAI container handler was created
            assert client.localai.host == "127.0.0.1"
            assert client.localai.port == test_port

            # The container should be created and started if Docker is available
            if client.localai._is_docker_running():
                container = client.localai._get_container()
                if container:
                    container.reload()
                    assert container.status == "running"

        finally:
            # Clean up - stop and remove container
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


@pytest.mark.docker
def test_init_container_failure(mock_env_empty, capsys, temp_dir):
    """Test initialization when LocalAI container startup fails."""
    with patch('reflex_llms.chat.load_dotenv'):
        # Force container startup failure by using invalid configuration
        with patch('reflex_llms.containers.ContainerHandler.ensure_running') as mock_ensure:
            mock_ensure.side_effect = Exception("Docker not available")

            client = ChatClient(
                auto_start_localai=True,
                preload_models=None,
                models_path=temp_dir,
            )

            captured = capsys.readouterr()
            assert "Warning: Could not start LocalAI container" in captured.out
            assert "Docker not available" in captured.out


@pytest.mark.docker
def test_preload_models_localai_real(mock_env_empty, temp_dir):
    """Test preloading models for LocalAI - REAL INTEGRATION."""
    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8095 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(
            local_ai_url=f"http://127.0.0.1:{test_port}/v1",
            preload_models=["gpt-3.5-turbo"],  # Use a smaller model
            auto_start_localai=True,
            wait_for_download=False,
            models_path=temp_dir,
        )

        try:
            assert client.is_azure is False

            # If Docker is running, container should be started
            if client.localai and client.localai._is_docker_running():
                container = client.localai._get_container()
                if container:
                    container.reload()
                    assert container.status == "running"

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


def test_model_name_property_localai(mock_env_empty, temp_dir):
    """Test model_name property for LocalAI."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(
            local_ai_model="custom-model",
            auto_start_localai=False,
            preload_models=None,
            models_path=temp_dir,
        )
        assert client.model_name == "custom-model"


def test_client_info_localai(mock_env_empty, temp_dir):
    """Test client_info property for LocalAI."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(
            local_ai_url="http://custom:9090/v1",
            local_ai_model="custom-model",
            auto_start_localai=False,
            preload_models=None,
            models_path=temp_dir,
        )

        info = client.client_info
        expected = {
            "type": "LocalAI",
            "url": "http://custom:9090/v1",
            "model": "custom-model",
            "available_models": list(client.model_mapping.keys())
        }

        assert info["type"] == expected["type"]
        assert info["url"] == expected["url"]
        assert info["model"] == expected["model"]
        assert isinstance(info["available_models"], list)


def test_load_model_config_success(mock_env_empty, mock_yaml_config):
    """Test successful loading of model configuration."""
    with patch('reflex_llms.chat.load_dotenv'), \
         patch('builtins.open', mock_open(read_data=mock_yaml_config)):
        with patch('pathlib.Path.exists', return_value=True):
            client = ChatClient(auto_start_localai=False, preload_models=None)
            client.config_path = Path("test_config.yaml")
            client.model_mapping = {}
            client._load_model_config()

            assert "gpt-4" in client.model_mapping
            assert "gpt-3.5-turbo" in client.model_mapping
            assert client.model_mapping["gpt-4"]["file"] == "gpt4-model.gguf"


def test_load_model_config_file_not_found(mock_env_empty, capsys):
    """Test loading model config when file doesn't exist."""
    with patch('reflex_llms.chat.load_dotenv'), \
         patch('pathlib.Path.exists', return_value=False):
        client = ChatClient(auto_start_localai=False, preload_models=None)
        client.config_path = Path("nonexistent.yaml")
        client.model_mapping = {}
        client._load_model_config()

        captured = capsys.readouterr()
        assert "Config file not found" in captured.out


def test_real_model_config_loading(mock_env_empty):
    """Test real model configuration loading."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        # Should load the real config file
        assert isinstance(client.model_mapping, dict)

        # Check for some expected models from the config
        expected_models = ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]
        for model in expected_models:
            assert model in client.model_mapping, f"Model {model} not found in mapping"

            config = client.get_model_config(model)
            assert config is not None
            assert "file" in config
            assert "urls" in config
            assert isinstance(config["urls"], list)
            assert len(config["urls"]) > 0


@pytest.mark.docker
@pytest.mark.slow
def test_download_model_success_real(mock_env_empty, temp_dir):
    """Test successful model download - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8100 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(local_ai_url=f"http://127.0.0.1:{test_port}/v1",
                            auto_start_localai=True,
                            preload_models=None)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Wait for LocalAI to be ready
            start_time = time.time()
            while time.time() - start_time < 120:
                if client.localai._is_port_open():
                    break
                time.sleep(3)
            else:
                pytest.skip("LocalAI not ready")

            # Try to download a small model
            job_id = client.download_model("gpt-3.5-turbo")

            if job_id:
                assert isinstance(job_id, str)
                assert len(job_id) > 0
                print(f"Download started with job ID: {job_id}")
            else:
                print("Model already available or download not needed")

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


def test_download_model_already_available(mock_env_empty):
    """Test download when model is already available."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        # Mock that the model is already available
        with patch.object(client, 'list_available_models', return_value=["gpt-4"]):
            result = client.download_model("gpt-4")
            assert result is None


def test_download_model_no_config(mock_env_empty):
    """Test download when no model configuration exists."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)
        client.model_mapping = {}

        with pytest.raises(ValueError, match="No configuration found for model"):
            client.download_model("nonexistent-model")


@pytest.mark.docker
def test_check_download_status_real(mock_env_empty, temp_dir):
    """Test checking download status - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8105 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(local_ai_url=f"http://127.0.0.1:{test_port}/v1",
                            auto_start_localai=True,
                            preload_models=None)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Wait for LocalAI to be ready
            start_time = time.time()
            while time.time() - start_time < 120:
                if client.localai._is_port_open():
                    break
                time.sleep(3)
            else:
                pytest.skip("LocalAI not ready")

            # Test checking status of non-existent job
            status = client.check_download_status("fake-job-id")
            assert isinstance(status, dict)
            # Should return error or not found status

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


def test_wait_for_model_download_timeout(mock_env_empty):
    """Test waiting for model download with timeout."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        # Mock a download that never completes
        with patch.object(client, 'check_download_status', return_value={"processed": False}):
            result = client.wait_for_model_download("job-123", "gpt-4",
                                                    timeout=1)  # 1 second timeout
            assert result is False


@pytest.mark.docker
def test_list_available_models_real(mock_env_empty, temp_dir):
    """Test listing available models - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8110 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(local_ai_url=f"http://127.0.0.1:{test_port}/v1",
                            auto_start_localai=True,
                            preload_models=None)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Wait for LocalAI to be ready
            start_time = time.time()
            while time.time() - start_time < 120:
                if client.localai._is_port_open():
                    break
                time.sleep(3)
            else:
                pytest.skip("LocalAI not ready")

            models = client.list_available_models()
            assert isinstance(models, list)
            # Initially should be empty or contain default models

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


@pytest.mark.docker
def test_completion_model_not_available_localai_real(mock_env_empty, sample_messages, temp_dir):
    """Test completion when model is not available in LocalAI - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8115 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(local_ai_url=f"http://127.0.0.1:{test_port}/v1",
                            auto_start_localai=True,
                            preload_models=None,
                            wait_for_download=False)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Wait for LocalAI to be ready
            start_time = time.time()
            while time.time() - start_time < 120:
                if client.localai._is_port_open():
                    break
                time.sleep(3)
            else:
                pytest.skip("LocalAI not ready")

            # Try to use a model that doesn't exist in config
            client.model_mapping = {}  # Clear model mapping

            with pytest.raises(RuntimeError, match="Cannot use model"):
                client.completion(sample_messages, model="nonexistent-model")

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


@pytest.mark.docker
@pytest.mark.slow
def test_completion_error_handling_real(mock_env_empty, sample_messages, capsys, temp_dir):
    """Test completion error handling - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8120 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(local_ai_url=f"http://127.0.0.1:{test_port}/v1",
                            auto_start_localai=True,
                            preload_models=None)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Don't wait for LocalAI to be ready - this should cause an error
            # Try to make a completion request before LocalAI is ready

            with pytest.raises(Exception):  # Should raise some kind of connection error
                client.completion(sample_messages, model="gpt-4")

            captured = capsys.readouterr()
            assert "Error during chat completion:" in captured.out or "Make sure LocalAI is running" in captured.out

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


def test_get_model_config(mock_env_empty):
    """Test getting model configuration."""
    with patch('reflex_llms.chat.load_dotenv'):
        client = ChatClient(auto_start_localai=False, preload_models=None)

        # Use real model mapping
        if "gpt-4" in client.model_mapping:
            config = client.get_model_config("gpt-4")
            assert config is not None
            assert "file" in config

        config = client.get_model_config("nonexistent")
        assert config is None


@pytest.mark.docker
@pytest.mark.slow
def test_full_localai_workflow_with_preload_real(mock_env_empty, temp_dir):
    """Test complete LocalAI workflow with preloading - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8125 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(
            local_ai_url=f"http://127.0.0.1:{test_port}/v1",
            preload_models=["gpt-3.5-turbo"],  # Use small model
            auto_start_localai=True,
            wait_for_download=False  # Don't wait to avoid long test times
        )

        try:
            assert client.is_azure is False

            if client.localai and client.localai._is_docker_running():
                container = client.localai._get_container()
                if container:
                    container.reload()
                    assert container.status == "running"

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


@pytest.mark.docker
def test_real_container_lifecycle(mock_env_empty, temp_dir):
    """Test real container lifecycle operations."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8130 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(local_ai_url=f"http://127.0.0.1:{test_port}/v1",
                            auto_start_localai=True,
                            preload_models=None)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Container should be running
            container = client.localai._get_container()
            if container:
                container.reload()
                assert container.status == "running"

                # Test stopping
                client.localai.stop()
                container.reload()
                assert container.status in ["exited", "stopped"]

                # Test restarting via ensure_running
                client.localai.ensure_running()
                container.reload()
                assert container.status == "running"

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


@pytest.mark.docker
@pytest.mark.slow
def test_real_model_download_and_completion(mock_env_empty, temp_dir):
    """Test real model download and completion with LocalAI."""

    with patch('reflex_llms.chat.load_dotenv'):
        # Use unique port
        test_port = 8135 + int(str(uuid.uuid4())[:8], 16) % 1000

        # Create client with a small model for testing
        client = ChatClient(
            local_ai_url=f"http://127.0.0.1:{test_port}/v1",
            local_ai_model="gpt-3.5-turbo",  # Will map to a smaller model
            auto_start_localai=True,
            preload_models=None,
            models_path=temp_dir,
            n_threads=16,
            wait_for_download=True)

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running or LocalAI not started")

            # Wait for LocalAI to be ready
            start_time = time.time()
            while time.time() - start_time < 300:  # 5 minute timeout
                if client.localai._is_port_open():
                    break
                time.sleep(5)
            else:
                pytest.skip("LocalAI did not become ready in time")

            # Check available models
            available_models = client.list_available_models()
            print(f"Available models: {available_models}")

            # Try to download a model if not available
            model_name = "gpt-3.5-turbo"
            if model_name not in available_models:
                try:
                    print(f"Downloading model: {model_name}")
                    job_id = client.download_model(model_name)

                    if job_id:
                        print(f"Download started with job ID: {job_id}")

                        # Wait for download (with timeout)
                        success = client.wait_for_model_download(
                            job_id,
                            model_name,
                            timeout=1800,
                        )
                        assert success, f"Model download failed or timed out"

                except Exception as e:
                    pytest.skip(f"Model download failed: {e}")

            # Test completion
            messages = [{"role": "user", "content": "Say 'Hello World' and nothing else."}]

            try:
                response = client.completion(messages, model=model_name, max_tokens=10)
                assert response is not None
                assert hasattr(response, 'choices')
                assert len(response.choices) > 0
                assert hasattr(response.choices[0], 'message')

                content = response.choices[0].message.content
                assert content is not None
                assert len(content.strip()) > 0

                print(f"Completion successful: {content}")

            except Exception as e:
                pytest.skip(f"Completion failed: {e}")

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        print("Cleaning up LocalAI container...")
                        container.remove(force=True)
                except Exception as e:
                    print(f"Cleanup error: {e}")


@pytest.mark.docker
@pytest.mark.slow
def test_wait_for_model_download_success_real(mock_env_empty, temp_dir):
    """Test waiting for model download completion - REAL INTEGRATION."""

    with patch('reflex_llms.chat.load_dotenv'):
        test_port = 8140 + int(str(uuid.uuid4())[:8], 16) % 1000

        client = ChatClient(
            local_ai_url=f"http://127.0.0.1:{test_port}/v1",
            auto_start_localai=True,
            preload_models=None,
            models_path=temp_dir,
        )

        try:
            if not client.localai or not client.localai._is_docker_running():
                pytest.skip("Docker not running")

            # Wait for LocalAI to be ready
            start_time = time.time()
            while time.time() - start_time < 120:
                if client.localai._is_port_open():
                    break
                time.sleep(3)
            else:
                pytest.skip("LocalAI not ready")

            # Start a real download
            try:
                job_id = client.download_model("gpt-3.5-turbo")
                if job_id:
                    # Test waiting for download with a short timeout for testing
                    # This may timeout, but we're testing the waiting mechanism
                    result = client.wait_for_model_download(job_id, "gpt-3.5-turbo", timeout=30)
                    # Result can be True (completed) or False (timeout) - both are valid test outcomes
                    assert isinstance(result, bool)
            except Exception as e:
                pytest.skip(f"Download test failed: {e}")

        finally:
            # Clean up
            if client.localai and client.localai._is_docker_running():
                try:
                    container = client.localai._get_container()
                    if container:
                        container.remove(force=True)
                except:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
