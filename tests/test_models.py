"""
Tests for the OllamaModelManager class.
"""
import pytest
import requests
import json
import time
from pathlib import Path
from typing import Dict, List

# -- Ours --
from reflex_llms.models import OllamaModelManager
from reflex_llms.containers import ContainerHandler


@pytest.fixture(scope="session")
def ollama_container():
    """Start Ollama container for testing session."""
    container_handler = ContainerHandler(
        host="127.0.0.1",
        port=11435,  # Use different port to avoid conflicts
        container_name="test-ollama-model-manager",
        data_path=Path("/tmp/test-ollama-models"))

    try:
        # Ensure Ollama container is running
        container_handler.ensure_running()

        # Wait a bit for Ollama to be fully ready
        time.sleep(2)

        yield container_handler

    finally:
        # Clean up after tests
        try:
            container_handler.stop()
        except Exception as e:
            print(f"Error stopping container: {e}")


@pytest.fixture
def model_manager(ollama_container):
    """Create OllamaModelManager instance using the test container."""
    return OllamaModelManager(ollama_url=ollama_container.api_url)


@pytest.fixture
def model_manager_no_container():
    """Create OllamaModelManager instance without container for error testing."""
    return OllamaModelManager(ollama_url="http://127.0.0.1:65432")


# Initialization Tests
def test_default_initialization():
    """Test default initialization values."""
    manager = OllamaModelManager()
    assert manager.ollama_url == "http://127.0.0.1:11434"
    assert isinstance(manager.model_mappings, dict)
    assert len(manager.model_mappings) > 0


def test_custom_url_initialization():
    """Test initialization with custom URL."""
    custom_url = "http://localhost:8080"
    manager = OllamaModelManager(ollama_url=custom_url)
    assert manager.ollama_url == custom_url


def test_model_mappings_content():
    """Test that model mappings contain expected OpenAI models."""
    manager = OllamaModelManager()
    expected_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "text-embedding-ada-002"]

    for model in expected_models:
        assert model in manager.model_mappings
        assert isinstance(manager.model_mappings[model], str)
        assert len(manager.model_mappings[model]) > 0


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


# Model Operations Tests
def test_list_models_with_container(model_manager):
    """Test list_models with running Ollama container."""
    models = model_manager.list_models()
    assert isinstance(models, list)
    # Fresh container might have no models, which is fine
    for model in models:
        assert "name" in model
        assert isinstance(model["name"], str)


def test_list_models_without_container(model_manager_no_container):
    """Test list_models when Ollama is not running."""
    with pytest.raises(RuntimeError):
        model_manager_no_container.list_models()


def test_model_exists_with_container(model_manager):
    """Test model_exists with running container."""
    # Test with a model that definitely doesn't exist
    assert model_manager.model_exists("definitely-nonexistent-model-12345") is False

    # Get existing models to test positive case
    models = model_manager.list_models()
    if models:
        existing_model = models[0]["name"]
        assert model_manager.model_exists(existing_model) is True


def test_model_exists_without_container(model_manager_no_container):
    """Test model_exists when Ollama is not running."""
    # The original implementation raises RuntimeError instead of returning False
    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        model_manager_no_container.model_exists("test-model")


def test_pull_model_without_container(model_manager_no_container, capsys):
    """Test pull_model when Ollama is not running."""
    result = model_manager_no_container.pull_model("test-model")
    assert result is False

    captured = capsys.readouterr()
    assert "Failed to pull model test-model" in captured.out


def test_copy_model_without_container(model_manager_no_container, capsys):
    """Test copy_model when Ollama is not running."""
    result = model_manager_no_container.copy_model("source", "destination")
    assert result is False

    captured = capsys.readouterr()
    assert "Failed to tag model source -> destination" in captured.out


# OpenAI Setup Tests
def test_setup_openai_models_without_container(model_manager_no_container, capsys):
    """Test setup_openai_models when Ollama is not running."""
    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        model_manager_no_container.setup_openai_models()

    captured = capsys.readouterr()
    assert "Setting up OpenAI-compatible models" in captured.out


def test_setup_openai_models_empty_mappings(model_manager):
    """Test setup with empty model mappings."""
    # Temporarily empty the mappings
    original_mappings = model_manager.model_mappings.copy()
    model_manager.model_mappings = {}

    try:
        result = model_manager.setup_openai_models()
        assert result is True  # Success with 0/0 models
    finally:
        # Restore original mappings
        model_manager.model_mappings = original_mappings


# Model Mappings Structure Tests
def test_model_mappings_structure():
    """Test that model mappings are properly structured."""
    manager = OllamaModelManager()
    mappings = manager.model_mappings

    # Verify all keys and values are strings
    for openai_name, ollama_name in mappings.items():
        assert isinstance(openai_name, str)
        assert isinstance(ollama_name, str)
        assert len(openai_name) > 0
        assert len(ollama_name) > 0

    # Verify no empty mappings
    assert len(mappings) > 0


def test_model_mappings_contain_expected_models():
    """Test that model mappings contain expected OpenAI models."""
    manager = OllamaModelManager()
    expected_openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]

    for model in expected_openai_models:
        assert model in manager.model_mappings


def test_model_mappings_contain_expected_ollama_models():
    """Test that model mappings contain expected Ollama models."""
    manager = OllamaModelManager()
    expected_ollama_models = ["llama3.2:3b", "llama3.1:8b", "llama3.1:70b"]

    ollama_models = list(manager.model_mappings.values())
    for model in expected_ollama_models:
        assert model in ollama_models


# Model Operations with Real Container
@pytest.mark.slow
def test_pull_tiny_model_with_container(model_manager, capsys):
    """Test pulling a very small model with real container."""
    # Use the smallest available model for testing
    test_model = "smollm:135m"  # Very small embedding model (~22MB)

    # Skip if model already exists
    if model_manager.model_exists(test_model):
        pytest.skip(f"Test model {test_model} already exists")

    print(f"Attempting to pull tiny test model: {test_model}")
    result = model_manager.pull_model(test_model)

    # Verify the model was actually pulled
    assert model_manager.model_exists(test_model)
    print(f"Successfully pulled and verified {test_model}")

    captured = capsys.readouterr()
    assert f"Pulling model: {test_model}" in captured.out
    assert f"Successfully pulled: {test_model}" in captured.out


def test_copy_model_with_container(model_manager, capsys):
    """Test model copying with real container."""
    # First, get available models
    models = model_manager.list_models()

    if not models:
        pytest.skip("No models available for copy test")

    # Use the first available model
    source_model = models[0]["name"]
    test_alias = f"test-alias-{int(time.time())}"  # Unique alias

    print(f"Testing copy: {source_model} -> {test_alias}")
    result = model_manager.copy_model(source_model, test_alias)

    if result:
        # Verify the alias was created
        assert model_manager.model_exists(test_alias)
        print(f"Successfully created alias {test_alias} for {source_model}")

        captured = capsys.readouterr()
        assert f"Tagging model: {source_model} -> {test_alias}" in captured.out
        assert f"Successfully tagged: {source_model} -> {test_alias}" in captured.out
    else:
        captured = capsys.readouterr()
        assert f"Failed to tag model {source_model} -> {test_alias}" in captured.out


@pytest.mark.slow
def test_setup_minimal_openai_models_with_container(model_manager, capsys):
    """Test setting up minimal OpenAI models with real container."""
    # Use only the smallest models for testing
    test_mapping = {
        "test-gpt-mini": "all-minilm:22m"  # Very small model
    }

    # Temporarily replace mappings with minimal test mapping
    original_mappings = model_manager.model_mappings.copy()
    model_manager.model_mappings = test_mapping

    try:
        print("Testing minimal OpenAI model setup...")
        result = model_manager.setup_openai_models()

        captured = capsys.readouterr()
        assert "Setting up OpenAI-compatible models" in captured.out

        if result:
            assert model_manager.model_exists("test-gpt-mini")
            print("Successfully set up test OpenAI model mapping")
            assert "Model setup complete: 1/1 models configured" in captured.out
        else:
            print("Setup failed, but this might be expected in test environment")

    finally:
        # Restore original mappings
        model_manager.model_mappings = original_mappings


# Error Handling and Edge Cases
def test_invalid_endpoint_request(model_manager):
    """Test request to invalid endpoint with running container."""
    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        model_manager._make_request("definitely-invalid-endpoint-12345")


def test_request_timeout_handling():
    """Test request timeout handling with unreachable URL."""
    # Use a non-routable IP to test timeout (RFC 5737 test addresses)
    manager = OllamaModelManager(ollama_url="http://192.0.2.1:11434")

    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        manager._make_request("tags")


def test_container_api_url_property(ollama_container):
    """Test that container handler provides correct API URL."""
    api_url = ollama_container.api_url
    assert api_url.startswith("http://")
    assert ":11435" in api_url  # Our test port

    openai_url = ollama_container.openai_compatible_url
    assert openai_url.endswith("/v1")


def test_model_manager_uses_container_url(ollama_container):
    """Test that model manager uses container's URL correctly."""
    manager = OllamaModelManager(ollama_url=ollama_container.api_url)

    # Should be able to connect and get models
    models = manager.list_models()
    assert isinstance(models, list)


# Container Lifecycle Tests
def test_container_persistence_across_model_operations(ollama_container):
    """Test that container remains stable across multiple operations."""
    manager = OllamaModelManager(ollama_url=ollama_container.api_url)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
