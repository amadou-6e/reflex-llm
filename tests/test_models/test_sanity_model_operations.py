"""List models and model existence sanity tests
"""
import pytest

# -- Ours --
from reflex_llms.models import OllamaManager
# -- Tests --
from tests.test_models import *


# Model Operations Tests
def test_list_models_with_container(model_manager: OllamaManager):
    """Test list_models with running Ollama container."""
    models = model_manager.list_models()
    assert isinstance(models, list)
    # Fresh container might have no models, which is fine
    for model in models:
        assert "name" in model
        assert isinstance(model["name"], str)


def test_list_models_without_container(model_manager_no_container: OllamaManager):
    """Test list_models when Ollama is not running."""
    with pytest.raises(RuntimeError):
        model_manager_no_container.list_models()


def test_model_exists_with_container(model_manager: OllamaManager):
    """Test model_exists with running container."""
    # Test with a model that definitely doesn't exist
    assert model_manager.model_exists("definitely-nonexistent-model-12345") is False

    # Get existing models to test positive case
    models = model_manager.list_models()
    if models:
        existing_model = models[0]["name"]
        assert model_manager.model_exists(existing_model) is True


def test_model_exists_without_container(model_manager_no_container: OllamaManager):
    """Test model_exists when Ollama is not running."""
    # The original implementation raises RuntimeError instead of returning False
    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        model_manager_no_container.model_exists("test-model")
