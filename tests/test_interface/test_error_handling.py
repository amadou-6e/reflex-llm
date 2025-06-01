import pytest
import os
from unittest.mock import patch
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms

# -- Tests --
from tests.conftest import *
from tests.test_interface import *


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
