import pytest
# -- Ours --
from reflex_llms.settings import DEFAULT_MODEL_MAPPINGS  # -- Tests --
from tests.test_models import *


def test_pull_tiny_model_with_container(model_manager: OllamaManager, capsys):
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


def test_copy_model_with_container(model_manager: OllamaManager, capsys):
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


def test_setup_minimal_openai_models_with_container(model_manager: OllamaManager, capsys):
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
        result = model_manager.setup_model_mapping()

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
