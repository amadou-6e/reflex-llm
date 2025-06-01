import pytest
import os

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms

# -- Tests --
from tests.conftest import *


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
