from reflex_llms.models import OllamaManager
from tests.test_models import *


# Container Integration Tests
def test_container_handler_starts_ollama(ollama_container):
    """Test that container handler successfully starts Ollama."""
    assert ollama_container._is_port_open()
    assert ollama_container._is_container_running()


def test_model_manager_connects_to_container(model_manager):
    """Test that model manager can connect to containerized Ollama."""
    # This should not raise an exception
    models = model_manager.list_models()
    assert isinstance(models, list)


def test_container_api_url_property(ollama_container):
    """Test that container handler provides correct API URL."""
    api_url = ollama_container.api_url
    assert api_url.startswith("http://")
    assert ":11435" in api_url  # Our test port

    openai_url = ollama_container.openai_compatible_url
    assert openai_url.endswith("/v1")


def test_model_manager_uses_container_url(ollama_container):
    """Test that model manager uses container's URL correctly."""
    manager = OllamaManager(ollama_url=ollama_container.api_url)

    # Should be able to connect and get models
    models = manager.list_models()
    assert isinstance(models, list)


# Container Lifecycle Tests
def test_container_persistence_across_model_operations(ollama_container):
    """Test that container remains stable across multiple operations."""
    manager = OllamaManager(ollama_url=ollama_container.api_url)

    # Multiple operations should all work
    models1 = manager.list_models()
    assert isinstance(models1, list)

    # Container should still be running
    assert ollama_container._is_container_running()
    assert ollama_container._is_port_open()

    models2 = manager.list_models()
    assert isinstance(models2, list)

    # Results should be consistent
    assert len(models1) == len(models2)
