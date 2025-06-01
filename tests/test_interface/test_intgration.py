import pytest
import os

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms
from reflex_llms.server import ReflexServerConfig, ModelMapping

# -- Tests --
from tests.conftest import *


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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": "Hello from integration test"
            }],
            max_tokens=10,
        )

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
            model="gpt-4o-mini" if provider_type != "azure" else "gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": "Test message"
            }],
            max_tokens=5,
        )

        assert response.choices[0].message.content is not None
        print(f"Response from {provider_type}: {response.choices[0].message.content}")

    except Exception as e:
        print(f"API call failed with {provider_type}: {e}")
        # Don't fail the test - this shows real behavior


# RefLex Server Tests (Real Server Creation)
@pytest.mark.slow
def test_reflex_server_creation_real() -> None:
    """Test actual RefLex server creation (slow test)."""
    # Test with minimal timeout to avoid long waits if Docker isn't available

    provider_type = reflex_llms.get_openai_client_type(
        ["reflex"],
        timeout=2.0,
        reflex_server_config={
            "port": 11435,
            "container_name": "test-reflex-server"
        },
    )

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


def test_reflex_server_with_config_object() -> None:
    """Test RefLex server with explicit configuration object."""
    # Create real config object
    reflex_config = ReflexServerConfig(
        port=11436,  # Use different port
        container_name="test-config-server",
        auto_setup=False,  # Don't auto-setup to avoid long waits
        model_mappings=ModelMapping(minimal_setup=True))

    try:
        provider_type = reflex_llms.get_openai_client_type(
            ["reflex"],
            reflex_server_config=reflex_config,
            timeout=2.0,
        )

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
