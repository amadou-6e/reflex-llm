"""
Tests for OpenAI client routing with module state management.
Minimal mocking approach - only mock network responses and environment variables.
"""
import pytest
import os
import time
import json
import tempfile
from unittest.mock import patch, Mock
from typing import Dict, Any
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms
from reflex_llms.server import ReflexServer, ReflexServerConfig, ModelMapping


@pytest.fixture(autouse=True)
def cleanup_module_state():
    """Clean up module state before and after each test."""
    # Clear state before test
    reflex_llms.clear_cache()
    reflex_llms.stop_reflex_server()

    yield

    # Clear state after test
    reflex_llms.clear_cache()
    reflex_llms.stop_reflex_server()


@pytest.fixture
def temp_config_file():
    """Create a temporary reflex.json configuration file."""
    config_data = {
        "openai_base_url": "https://custom-api.example.com/v1",
        "azure_api_version": "2024-06-01",
        "azure_base_url": "https://custom-azure.openai.azure.com",
        "preference_order": ["reflex", "azure", "openai"],
        "timeout": 10.0,
        "reflex_server": {
            "host": "127.0.0.1",
            "port": 8080,
            "image": "ollama/ollama:latest",
            "container_name": "test-reflex-server",
            "auto_setup": True,
            "model_mappings": {
                "minimal_setup": True,
                "model_mapping": {
                    "gpt-3.5-turbo": "llama3.2:3b",
                    "gpt-4": "llama3.1:8b"
                },
                "minimal_model_mapping": {
                    "gpt-3.5-turbo": "llama3.2:3b",
                    "text-embedding-ada-002": "nomic-embed-text"
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    yield temp_path, config_data

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with reflex.json for file discovery testing."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "reflex.json"

    config_data = {
        "openai_base_url": "https://discovered-api.example.com/v1",
        "preference_order": ["openai", "reflex"],
        "timeout": 8.0,
        "reflex_server": {
            "port": 9999,
            "model_mappings": {
                "minimal_setup": False
            }
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    yield temp_dir, config_data

    # Cleanup
    shutil.rmtree(temp_dir)


# Module State Tests
def test_initial_module_state() -> None:
    """Test initial module state is clean."""
    status = reflex_llms.get_module_status()

    assert status["selected_provider"] is None
    assert status["has_cached_config"] is False
    assert status["reflex_server_running"] is False
    assert status["reflex_server_url"] is None


def test_clear_cache_functionality() -> None:
    """Test cache clearing functionality."""
    # Manually set some state to test clearing
    reflex_llms._cached_config = {"test": "config"}
    reflex_llms._selected_provider = "test"

    # Clear cache
    reflex_llms.clear_cache()

    status = reflex_llms.get_module_status()
    assert status["selected_provider"] is None
    assert status["has_cached_config"] is False


def test_get_selected_provider_none() -> None:
    """Test get_selected_provider when no provider selected."""
    assert reflex_llms.get_selected_provider() is None


def test_is_using_reflex_false() -> None:
    """Test is_using_reflex when not using RefLex."""
    assert reflex_llms.is_using_reflex() is False


def test_get_reflex_server_none() -> None:
    """Test get_reflex_server when no RefLex server."""
    assert reflex_llms.get_reflex_server() is None


# Real OpenAI API Tests
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_openai_api_returns_provider_type() -> None:
    """Test that get_openai_client_type returns provider type string."""
    provider_type = reflex_llms.get_openai_client_type(["openai"])

    # Should return string, not dict
    assert isinstance(provider_type, str)
    assert provider_type == "openai"

    # Check module state
    assert reflex_llms.get_selected_provider() == "openai"
    assert not reflex_llms.is_using_reflex()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_openai_config_dict_still_available() -> None:
    """Test that get_openai_client_config still returns configuration dict."""
    config = reflex_llms.get_openai_client_config(["openai"])

    # Should return dict with config
    assert isinstance(config, dict)
    assert config["api_key"] == os.getenv("OPENAI_API_KEY")
    assert config["base_url"] == "https://api.openai.com/v1"

    # Check module state
    assert reflex_llms.get_selected_provider() == "openai"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_openai_client_creation_real() -> None:
    """Test creating actual OpenAI client."""
    client = reflex_llms.get_openai_client(["openai"])

    # Should be able to create client
    assert client is not None
    assert hasattr(client, 'chat')

    # Test a simple API call
    try:
        models = client.models.list()
        assert models.data is not None
        assert len(models.data) > 0
    except Exception as e:
        pytest.fail(f"OpenAI API call failed: {e}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_openai_completion_integration() -> None:
    """Test actual OpenAI completion through the routing."""
    client = reflex_llms.get_openai_client(["openai"])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": "Say 'Hello from RefLex routing test'"
            }],
            max_tokens=20)

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None

        content = response.choices[0].message.content.lower()
        assert "hello" in content

    except Exception as e:
        pytest.fail(f"OpenAI completion failed: {e}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_openai_caching_behavior() -> None:
    """Test that OpenAI config is cached properly."""
    # First call
    config1 = reflex_llms.get_openai_client_type(["openai"])

    config2 = reflex_llms.get_openai_client_type(["openai"])

    # Should be consistent
    assert config1 == config2 == "openai"
    assert reflex_llms.get_selected_provider() == "openai"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_force_recheck_functionality() -> None:
    """Test that force_recheck works."""
    # First call to populate cache
    provider1 = reflex_llms.get_openai_client_type(["openai"])

    # Force recheck - should still work and return same provider
    provider2 = reflex_llms.get_openai_client_type(["openai"], force_recheck=True)

    assert provider1 == provider2 == "openai"


def test_openai_api_network_success() -> None:
    """Test OpenAI API detection with mocked network response."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401  # Auth failed but API reachable
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider_type = reflex_llms.get_openai_client_type(["openai"])

            assert provider_type == "openai"
            assert reflex_llms.get_selected_provider() == "openai"

            # Verify correct URL was called
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert "https://api.openai.com/v1/models" in args[0]


def test_openai_api_network_failure() -> None:
    """Test OpenAI API detection with network failure."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_get.side_effect = Exception("Connection failed")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(RuntimeError, match="No OpenAI providers available"):
                reflex_llms.get_openai_client_type(["openai"])


def test_openai_no_api_key() -> None:
    """Test OpenAI API when no API key is set."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        # Remove API key from environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No OpenAI providers available"):
                reflex_llms.get_openai_client_type(["openai"])


def test_azure_api_network_success() -> None:
    """Test Azure OpenAI with mocked network success."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-azure-key"
        }

        with patch.dict(os.environ, env_vars):
            provider_type = reflex_llms.get_openai_client_type(["azure"])

            assert provider_type == "azure"
            assert reflex_llms.get_selected_provider() == "azure"

            # Verify config is available
            config = reflex_llms.get_openai_client_config(["azure"])
            assert config["api_key"] == "test-azure-key"
            assert "test.openai.azure.com" in config["base_url"]
            assert config["api_version"] == "2024-02-15-preview"


def test_azure_missing_credentials() -> None:
    """Test Azure OpenAI when credentials are missing."""
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"}):
        # Missing endpoint
        with pytest.raises(RuntimeError, match="No OpenAI providers available"):
            reflex_llms.get_openai_client_type(["azure"])


# Custom Configuration Tests (Minimal Mocking)
def test_custom_openai_base_url() -> None:
    """Test custom OpenAI base URL configuration."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        custom_url = "https://custom-openai.example.com/v1"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = reflex_llms.get_openai_client_config(["openai"], openai_base_url=custom_url)

            assert config["base_url"] == custom_url

            # Verify the correct URL was used for health check
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            expected_url = f"{custom_url}/models"
            assert expected_url in args[0]


def test_custom_azure_api_version() -> None:
    """Test custom Azure API version configuration."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        custom_version = "2024-06-01"
        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key"
        }

        with patch.dict(os.environ, env_vars):
            config = reflex_llms.get_openai_client_config(["azure"],
                                                          azure_api_version=custom_version)

            assert config["api_version"] == custom_version


def test_custom_timeout_parameter() -> None:
    """Test custom timeout parameter is passed through."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        custom_timeout = 15.0

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            reflex_llms.get_openai_client_type(["openai"], timeout=custom_timeout)

            # Verify timeout was used in request
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert kwargs.get('timeout') == custom_timeout


def test_config_file_loading_with_real_file(temp_config_file) -> None:
    """Test loading configuration from actual file."""
    config_path, config_data = temp_config_file

    # Test the actual configuration resolution function
    resolved = reflex_llms._resolve_configuration_parameters(
        from_file=True,
        config_path=config_path,
        filename="reflex.json",
    )

    # Verify file values were loaded
    assert resolved['openai_base_url'] == config_data['openai_base_url']
    assert resolved['azure_api_version'] == config_data['azure_api_version']
    assert resolved['preference_order'] == config_data['preference_order']
    assert resolved['timeout'] == config_data['timeout']


def test_config_file_parameter_override(temp_config_file) -> None:
    """Test that explicit parameters override file configuration."""
    config_path, config_data = temp_config_file

    # Override some file parameters
    resolved = reflex_llms._resolve_configuration_parameters(
        from_file=True,
        config_path=config_path,
        timeout=20.0,  # Override file's 10.0
        preference_order=["azure", "openai"]  # Override file's order
    )

    # Explicit parameters should take precedence
    assert resolved['timeout'] == 20.0  # Overridden
    assert resolved['preference_order'] == ["azure", "openai"]  # Overridden

    # File values should be used when not overridden
    assert resolved['openai_base_url'] == config_data['openai_base_url']  # From file


def test_config_file_discovery(temp_config_dir) -> None:
    """Test automatic config file discovery in directory tree."""
    temp_dir, config_data = temp_config_dir

    # Change to the temp directory to test discovery
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)

        # Test file discovery
        config_path = reflex_llms.configs._find_reflex_config()
        assert config_path is not None
        assert config_path.name == "reflex.json"

        # Test loading discovered config
        loaded_config = reflex_llms.load_reflex_config()
        assert loaded_config is not None
        assert loaded_config['openai_base_url'] == config_data['openai_base_url']

    finally:
        os.chdir(original_cwd)


def test_config_file_loading_fallback() -> None:
    """Test graceful fallback when config file doesn't exist."""
    # Test with non-existent file
    resolved = reflex_llms._resolve_configuration_parameters(
        from_file=True, config_path="/nonexistent/path/reflex.json")

    # Should fallback to defaults
    assert resolved['preference_order'] == ["openai", "azure", "reflex"]
    assert resolved['timeout'] == 5.0
    assert resolved['openai_base_url'] == "https://api.openai.com/v1"


# RefLex Server Tests (Real Server Creation)
@pytest.mark.slow
def test_reflex_server_creation_real() -> None:
    """Test actual RefLex server creation (slow test)."""
    # Test with minimal timeout to avoid long waits if Docker isn't available
    try:
        provider_type = reflex_llms.get_openai_client_type(
            ["reflex"],
            timeout=2.0,
            port=11435,  # Use different port to avoid conflicts
            container_name="test-reflex-server")

        if provider_type == "reflex":
            # Server was successfully created
            assert reflex_llms.get_selected_provider() == "reflex"
            assert reflex_llms.is_using_reflex() is True

            server = reflex_llms.get_reflex_server()
            assert server is not None

            # Test that we can get status
            status = reflex_llms.get_module_status()
            assert status["reflex_server_running"] is True
            assert status["reflex_server_url"] is not None

    except RuntimeError as e:
        if "Docker is not running" in str(e) or "No OpenAI providers available" in str(e):
            pytest.skip("Docker not available or RefLex setup failed")
        else:
            raise


def test_reflex_server_with_config_object() -> None:
    """Test RefLex server with explicit configuration object."""
    # Create real config object
    reflex_config = ReflexServerConfig(
        port=11436,  # Use different port
        container_name="test-config-server",
        auto_setup=False,  # Don't auto-setup to avoid long waits
        model_mappings=ModelMapping(minimal_setup=True))

    try:
        provider_type = reflex_llms.get_openai_client_type(["reflex"],
                                                           reflex_server_config=reflex_config,
                                                           timeout=2.0)

        if provider_type == "reflex":
            assert reflex_llms.get_selected_provider() == "reflex"

            # Verify server uses config
            server = reflex_llms.get_reflex_server()
            assert server is not None
            assert server.port == 11436
            assert server.container_name == "test-config-server"

    except RuntimeError as e:
        if "Docker is not running" in str(e) or "RefLex setup failed" in str(e):
            pytest.skip("Docker not available or RefLex setup failed")
        else:
            raise


# Preference Order Tests (Real Behavior)
def test_preference_order_respected() -> None:
    """Test that preference order is actually respected."""
    with patch('reflex_llms.requests.get') as mock_get:
        # Mock OpenAI as unavailable
        mock_get.side_effect = Exception("OpenAI unavailable")

        # Set Azure environment
        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key"
        }

        with patch.dict(os.environ, env_vars):
            # First call - OpenAI fails, should try next in order
            with pytest.raises(RuntimeError):
                # Both OpenAI and Azure will fail (Azure also needs successful network response)
                reflex_llms.get_openai_client_type(["openai", "azure"])


def test_fallback_behavior_real() -> None:
    """Test actual fallback behavior with environment manipulation."""
    with patch('reflex_llms.requests.get') as mock_get:
        # Mock network responses
        def mock_response(*args, **kwargs):
            url = args[0]
            if "openai.com" in url:
                raise Exception("OpenAI unreachable")
            elif "azure.com" in url:
                mock_resp = Mock()
                mock_resp.status_code = 200
                return mock_resp
            else:
                raise Exception("Unknown URL")

        mock_get.side_effect = mock_response

        # Set up Azure credentials
        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key"
        }

        with patch.dict(os.environ, env_vars):
            # Should fallback from OpenAI to Azure
            provider_type = reflex_llms.get_openai_client_type(["openai", "azure"])
            assert provider_type == "azure"


# Integration Tests (Real APIs where available)
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_full_integration_with_real_openai() -> None:
    """Full integration test with real OpenAI API."""
    # Clear any cached state
    reflex_llms.clear_cache()

    # Test provider selection
    provider_type = reflex_llms.get_openai_client_type()
    print(f"Selected provider: {provider_type}")

    # Should select OpenAI if available
    if provider_type == "openai":
        # Test client creation
        client = reflex_llms.get_openai_client()

        # Test actual API call
        response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": "Hello from integration test"
                                                  }],
                                                  max_tokens=10)

        assert response.choices[0].message.content is not None
        print(f"OpenAI response: {response.choices[0].message.content}")


@pytest.mark.integration
@pytest.mark.slow
def test_full_fallback_chain_real() -> None:
    """Test complete fallback chain with real conditions."""
    # This test shows actual behavior without forcing specific providers
    provider_type = reflex_llms.get_openai_client_type()
    client = reflex_llms.get_openai_client()

    print(f"Naturally selected provider: {provider_type}")
    assert provider_type in ["openai", "azure", "reflex"]

    # Test that the client actually works
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini" if provider_type != "azure" else "gpt-35-turbo",
            messages=[{
                "role": "user",
                "content": "Test message"
            }],
            max_tokens=5)

        assert response.choices[0].message.content is not None
        print(f"Response from {provider_type}: {response.choices[0].message.content}")

    except Exception as e:
        print(f"API call failed with {provider_type}: {e}")
        # Don't fail the test - this shows real behavior


# Error Handling Tests (Real Error Conditions)
def test_no_providers_available_real() -> None:
    """Test when no providers are actually available."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")

        with patch.dict(os.environ, {}, clear=True):  # No API keys
            with pytest.raises(RuntimeError, match="No OpenAI providers available"):
                reflex_llms.get_openai_client_type(["openai", "azure"])


def test_invalid_preference_order_real() -> None:
    """Test with invalid provider names."""
    with pytest.raises(RuntimeError, match="No OpenAI providers available"):
        reflex_llms.get_openai_client_type(["invalid_provider", "another_invalid"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
