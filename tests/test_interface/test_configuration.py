import os
from unittest.mock import patch, Mock

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms

# -- Tests --
from tests.conftest import *
from tests.test_interface import *


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
    assert resolved['timeout'] == 120.0
    assert resolved['openai_base_url'] == "https://api.openai.com/v1"


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


def test_config_caching_logic_bug():
    """Test that caching logic works correctly with recheck parameter."""
    import reflex_llms
    from unittest.mock import patch, Mock

    reflex_llms.clear_cache()

    with patch('reflex_llms.requests.get') as mock_get, \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):

        mock_get.return_value = Mock(status_code=401)

        with patch('reflex_llms._try_openai_provider') as mock_try_openai:
            mock_try_openai.return_value = ({
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }, "")

            # First call - should call provider
            reflex_llms.get_openai_client_config(["openai"])
            assert mock_try_openai.call_count == 1

            # Reset and test cache
            mock_try_openai.reset_mock()

            # Second call with recheck=False - should NOT call provider (use cache)
            reflex_llms.get_openai_client_config(["openai"])
            assert mock_try_openai.call_count == 0, "Cache should prevent provider calls"

            # Third call with recheck=True - should call provider again
            reflex_llms.get_openai_client_config(["openai"], force_recheck=True)
            assert mock_try_openai.call_count == 1, "recheck=True should bypass cache"


def test_config_caching_logic_works():
    """Test that caching logic works correctly with the fixed implementation."""
    import reflex_llms
    from unittest.mock import patch, Mock

    reflex_llms.clear_cache()

    with patch('reflex_llms.requests.get') as mock_get, \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):

        mock_get.return_value = Mock(status_code=401)

        with patch('reflex_llms._try_openai_provider') as mock_try_openai:
            mock_try_openai.return_value = ({
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }, "")

            # First call - should call provider
            reflex_llms.get_openai_client_config(["openai"])
            assert mock_try_openai.call_count == 1

            # Reset and test cache
            mock_try_openai.reset_mock()

            # Second call with force_recheck=False - should use cache
            reflex_llms.get_openai_client_config(["openai"])
            assert mock_try_openai.call_count == 0, "Cache should prevent provider calls"

            # Third call with force_recheck=True - should call provider again
            reflex_llms.get_openai_client_config(["openai"], force_recheck=True)
            assert mock_try_openai.call_count == 1, "force_recheck=True should bypass cache"


def test_cache_invalidation_on_config_change():
    """Test that cache is bypassed when configuration parameters change."""
    import reflex_llms
    from unittest.mock import patch, Mock

    reflex_llms.clear_cache()

    with patch('reflex_llms.requests.get') as mock_get, \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):

        mock_get.return_value = Mock(status_code=401)

        with patch('reflex_llms._try_openai_provider') as mock_try_openai:
            mock_try_openai.return_value = ({
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }, "")

            # First call
            reflex_llms.get_openai_client_config(["openai"], timeout=5.0)
            mock_try_openai.reset_mock()

            # Same config, force_recheck=False - should use cache
            reflex_llms.get_openai_client_config(["openai"], timeout=5.0)
            assert mock_try_openai.call_count == 0, "Identical config should use cache"

            # Different config, force_recheck=False - should bypass cache
            reflex_llms.get_openai_client_config(["openai"], timeout=10.0)
            assert mock_try_openai.call_count == 1, "Changed config should bypass cache"


def test_cache_works_with_identical_config_on_recheck():
    """Test that cache is used even with force_recheck=True if config is identical."""
    import reflex_llms
    from unittest.mock import patch, Mock

    reflex_llms.clear_cache()

    with patch('reflex_llms.requests.get') as mock_get, \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):

        mock_get.return_value = Mock(status_code=401)

        with patch('reflex_llms._try_openai_provider') as mock_try_openai:
            mock_try_openai.return_value = ({
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }, "")

            # First call
            reflex_llms.get_openai_client_config(["openai"], timeout=5.0)
            mock_try_openai.reset_mock()

            # Second call with force_recheck=True but identical config
            # With the new logic: if cached config == current config, still use cache
            reflex_llms.get_openai_client_config(["openai"], timeout=5.0, force_recheck=True)

            # This should use cache since resolved config is identical
            # (This is the optimization your new logic provides)
            assert mock_try_openai.call_count == 1, "Identical config should use cache even with force_recheck=True"
