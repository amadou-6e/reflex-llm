"""Request handling sanity tests
"""
import pytest
from tests.test_models import *


# Request Handling Tests
def test_make_request_url_construction(model_manager):
    """Test URL construction for requests."""
    # This should work with the running container
    result = model_manager._make_request("tags")
    assert isinstance(result, dict)
    assert "models" in result


def test_make_request_with_invalid_endpoint(model_manager):
    """Test request with invalid endpoint."""
    with pytest.raises(RuntimeError):
        model_manager._make_request("invalid-endpoint-12345")


def test_make_request_without_container(model_manager_no_container):
    """Test request handling when Ollama is not available."""
    with pytest.raises(RuntimeError):
        model_manager_no_container._make_request("tags")


# Error Handling and Edge Cases
def test_invalid_endpoint_request(model_manager):
    """Test request to invalid endpoint with running container."""
    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        model_manager._make_request("definitely-invalid-endpoint-12345")


def test_request_timeout_handling():
    """Test request timeout handling with unreachable URL."""
    # Use a non-routable IP to test timeout (RFC 5737 test addresses)
    manager = OllamaManager(ollama_url="http://192.0.2.1:11434")

    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        manager._make_request("tags")
