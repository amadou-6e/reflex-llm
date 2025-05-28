"""
Tests for OpenAI client routing with module state management.
"""
import pytest
import os
import time
from unittest.mock import patch, Mock
from typing import Dict, Any
import tempfile
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms
from reflex_llms.server import ReflexServer
# -- Test the actual module --


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
def mock_requests_get():
    """Mock requests.get for testing network calls."""
    with patch('reflex_llms.requests.get') as mock:
        yield mock


@pytest.fixture
def mock_reflex_server():
    """Mock RefLex server for testing."""
    with patch('reflex_llms.ReflexServer') as mock:
        server_instance = Mock()
        server_instance._setup_complete = True
        server_instance.is_healthy = True
        server_instance.openai_compatible_url = "http://localhost:11434/v1"
        server_instance.stop = Mock()

        mock.return_value = server_instance
        yield mock, server_instance


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


# OpenAI API Tests (requires real credentials)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_openai_api_real_credentials() -> None:
    """Test OpenAI API with real credentials."""
    config = reflex_llms.get_openai_client_config(["openai"])

    assert config["api_key"] == os.getenv("OPENAI_API_KEY")
    assert config["base_url"] == "https://api.openai.com/v1"

    # Check module state
    assert reflex_llms.get_selected_provider() == "openai"
    assert not reflex_llms.is_using_reflex()


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
def test_openai_caching_behavior() -> None:
    """Test that OpenAI config is cached properly."""
    # First call
    config1 = reflex_llms.get_openai_client_config(["openai"])

    # Second call (should use cache)
    with patch('reflex_llms.requests.get') as mock_get:
        config2 = reflex_llms.get_openai_client_config(["openai"])

        # Should not have made network request (used cache)
        mock_get.assert_not_called()

        # Configs should be identical
        assert config1 == config2


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_force_recheck_bypasses_cache() -> None:
    """Test that force_recheck bypasses cache."""
    # First call to populate cache
    reflex_llms.get_openai_client_config(["openai"])

    # Force recheck should make network call
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        config = reflex_llms.get_openai_client_config(["openai"], force_recheck=True)

        # Should have made network request despite cache
        mock_get.assert_called_once()


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


# Azure OpenAI Tests (skipped - no credentials)
@pytest.mark.skip(reason="Azure OpenAI credentials not available")
def test_azure_openai_real_credentials() -> None:
    """Test Azure OpenAI with real credentials."""
    config = reflex_llms.get_openai_client_config(["azure"])

    assert "AZURE_OPENAI_ENDPOINT" in config["base_url"]
    assert config["api_key"] == os.getenv("AZURE_OPENAI_API_KEY")
    assert "api_version" in config


@pytest.mark.skip(reason="Azure OpenAI credentials not available")
def test_azure_client_creation() -> None:
    """Test creating Azure OpenAI client."""
    client = reflex_llms.get_openai_client(["azure"])

    # Test API call
    models = client.models.list()
    assert models.data is not None


@pytest.mark.skip(reason="Azure OpenAI credentials not available")
def test_azure_completion_integration() -> None:
    """Test Azure OpenAI completion."""
    client = reflex_llms.get_openai_client(["azure"])

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # Azure model name
        messages=[{
            "role": "user",
            "content": "Hello"
        }],
        max_tokens=10)
    assert response.choices[0].message.content is not None


# Mocked Provider Tests
def test_openai_api_mocked_success(mock_requests_get) -> None:
    """Test OpenAI API detection with mocked successful response."""
    # Mock successful OpenAI API response
    mock_response = Mock()
    mock_response.status_code = 401  # Auth failed but connected
    mock_requests_get.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        config = reflex_llms.get_openai_client_config(["openai"])

        assert config["api_key"] == "test-key"
        assert config["base_url"] == "https://api.openai.com/v1"
        assert reflex_llms.get_selected_provider() == "openai"


def test_openai_api_mocked_failure(mock_requests_get) -> None:
    """Test OpenAI API detection with mocked failure."""
    # Mock failed OpenAI API response
    mock_requests_get.side_effect = Exception("Connection failed")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with pytest.raises(RuntimeError, match="No OpenAI providers available"):
            reflex_llms.get_openai_client_config(["openai"])


def test_openai_no_api_key(mock_requests_get) -> None:
    """Test OpenAI API when no API key is set."""
    mock_response = Mock()
    mock_response.status_code = 401
    mock_requests_get.return_value = mock_response

    # Remove API key from environment
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="No OpenAI providers available"):
            reflex_llms.get_openai_client_config(["openai"])


def test_azure_mocked_success(mock_requests_get) -> None:
    """Test Azure OpenAI with mocked success."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-azure-key"
    }

    with patch.dict(os.environ, env_vars):
        config = reflex_llms.get_openai_client_config(["azure"])

        assert config["api_key"] == "test-azure-key"
        assert "test.openai.azure.com" in config["base_url"]
        assert config["api_version"] == "2024-02-15-preview"
        assert reflex_llms.get_selected_provider() == "azure"


def test_azure_missing_endpoint(mock_requests_get) -> None:
    """Test Azure OpenAI when endpoint is missing."""
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"}):
        with pytest.raises(RuntimeError, match="No OpenAI providers available"):
            reflex_llms.get_openai_client_config(["azure"])


# RefLex Server Tests
def test_reflex_server_mocked_success(mock_reflex_server) -> None:
    """Test RefLex server creation with mocked success."""
    mock_class, mock_instance = mock_reflex_server

    config = reflex_llms.get_openai_client_config(["reflex"])

    # Verify server was created
    mock_class.assert_called_once()

    # Verify config
    assert config["api_key"] == "reflex"
    assert config["base_url"] == "http://localhost:11434/v1"

    # Verify module state
    assert reflex_llms.get_selected_provider() == "reflex"
    assert reflex_llms.is_using_reflex() is True
    assert reflex_llms.get_reflex_server() == mock_instance


def test_reflex_server_reuse(mock_reflex_server) -> None:
    """Test that RefLex server is reused when cached."""
    mock_class, mock_instance = mock_reflex_server

    # First call - creates server
    config1 = reflex_llms.get_openai_client_config(["reflex"])

    # Second call - should reuse server
    config2 = reflex_llms.get_openai_client_config(["reflex"])

    # Should only create server once
    mock_class.assert_called_once()

    # Configs should be identical
    assert config1 == config2


def test_reflex_server_setup_failure(mock_reflex_server) -> None:
    """Test RefLex server setup failure."""
    mock_class, mock_instance = mock_reflex_server
    mock_instance._setup_complete = False  # Setup never completes

    with pytest.raises(RuntimeError, match="No OpenAI providers available"):
        reflex_llms.get_openai_client_config(["reflex"])

    # Should have attempted cleanup
    mock_instance.stop.assert_called()


def test_stop_reflex_server_functionality(mock_reflex_server) -> None:
    """Test stopping RefLex server."""
    mock_class, mock_instance = mock_reflex_server

    # Create RefLex server
    reflex_llms.get_openai_client_config(["reflex"])

    # Stop server
    reflex_llms.stop_reflex_server()

    # Should have called stop
    mock_instance.stop.assert_called()

    # Server should be None
    assert reflex_llms.get_reflex_server() is None


# Preference Order Tests
def test_preference_order_respected(mock_requests_get, mock_reflex_server) -> None:
    """Test that preference order is respected."""
    # Mock OpenAI as unavailable
    mock_requests_get.side_effect = Exception("OpenAI unavailable")

    # Set Azure environment (but will be tried after RefLex in this order)
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key"
    }

    with patch.dict(os.environ, env_vars):
        # Prefer RefLex over Azure
        config = reflex_llms.get_openai_client_config(["reflex", "azure"])

        # Should use RefLex (first in preference)
        assert reflex_llms.get_selected_provider() == "reflex"


def test_fallback_behavior(mock_requests_get, mock_reflex_server) -> None:
    """Test fallback when preferred providers fail."""
    # Mock OpenAI as unavailable
    mock_requests_get.side_effect = Exception("Network error")

    # Try OpenAI first, fallback to RefLex
    config = reflex_llms.get_openai_client_config(["openai", "reflex"])

    # Should fallback to RefLex
    assert reflex_llms.get_selected_provider() == "reflex"


# Convenience Function Tests
def test_dev_mode_preference_order(mock_reflex_server) -> None:
    """Test that dev mode prefers RefLex first."""
    client = reflex_llms.get_client_dev_mode()

    # Should use RefLex (first preference in dev mode)
    assert reflex_llms.get_selected_provider() == "reflex"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not found in environment")
def test_prod_mode_preference_order() -> None:
    """Test that prod mode prefers cloud APIs first."""
    client = reflex_llms.get_client_prod_mode()

    # Should use OpenAI (first preference in prod mode)
    assert reflex_llms.get_selected_provider() == "openai"


# Integration Tests
@pytest.mark.integration
@pytest.mark.slow
def test_real_reflex_server_integration() -> None:
    """Test with real RefLex server (slow test)."""
    config = reflex_llms.get_openai_client_config(["reflex"], timeout=10.0)

    # Should have created server
    assert reflex_llms.get_selected_provider() == "reflex"
    assert reflex_llms.is_using_reflex() is True

    server = reflex_llms.get_reflex_server()
    assert server is not None
    assert server.is_healthy is True

    # Test that we can create client and use it
    client = reflex_llms.get_openai_client(["reflex"])

    # Try a simple completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Say hello from RefLex integration test"
        }],
        max_tokens=15)

    assert response.choices[0].message.content is not None
    content = response.choices[0].message.content.lower()
    assert len(content) > 0

    print(f"RefLex completion response: {response.choices[0].message.content}")


@pytest.mark.integration
@pytest.mark.slow
def test_full_fallback_chain_integration() -> None:
    """Test the complete fallback chain in real conditions."""
    # Test default preference order with real conditions
    client = reflex_llms.get_openai_client()  # Default order

    selected = reflex_llms.get_selected_provider()
    print(f"Selected provider in fallback test: {selected}")

    # Should have selected something
    assert selected in ["openai", "azure", "reflex"]

    # Test that client works
    response = client.chat.completions.create(
        model="gpt-4o-mini" if selected != "azure" else "gpt-35-turbo",
        messages=[{
            "role": "user",
            "content": "Hello from fallback test"
        }],
        max_tokens=10)

    assert response.choices[0].message.content is not None


# Error Handling Tests
def test_no_providers_available() -> None:
    """Test when no providers are available."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")

        with patch.dict(os.environ, {}, clear=True):  # No API keys
            with pytest.raises(RuntimeError, match="No OpenAI providers available"):
                reflex_llms.get_openai_client_config(["openai", "azure"])


def test_invalid_preference_order() -> None:
    """Test with invalid provider in preference order."""
    with pytest.raises(RuntimeError, match="No OpenAI providers available"):
        reflex_llms.get_openai_client_config(["invalid_provider"])


def test_module_state_persistence() -> None:
    """Test that module state persists across function calls."""
    with patch('reflex_llms.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # First call
            config1 = reflex_llms.get_openai_client_config(["openai"])
            provider1 = reflex_llms.get_selected_provider()

            # Second call - should use cache
            config2 = reflex_llms.get_openai_client_config(["openai"])
            provider2 = reflex_llms.get_selected_provider()

            # Should be consistent
            assert config1 == config2
            assert provider1 == provider2 == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
