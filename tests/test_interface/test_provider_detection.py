import pytest
import os
from unittest.mock import patch, Mock

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms
from tests.test_interface import *


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
