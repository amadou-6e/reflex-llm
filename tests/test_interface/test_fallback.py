import pytest
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
