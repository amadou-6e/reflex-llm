"""Model mapping sanity tests
"""

# -- Ours --
from reflex_llms.models import OllamaManager
from reflex_llms.settings import DEFAULT_MINIMAL_MODEL_MAPPINGS
from reflex_llms.containers import DEFAULT_MODEL_MAPPINGS

# -- Tests --
from tests.test_models import *


# Model Mappings Structure Tests
def test_model_mappings_structure():
    """Test that model mappings are properly structured."""
    manager = OllamaManager()
    mappings = manager.model_mappings

    # Verify all keys and values are strings
    for openai_name, ollama_name in mappings.items():
        assert isinstance(openai_name, str)
        assert isinstance(ollama_name, str)
        assert len(openai_name) > 0
        assert len(ollama_name) > 0

    # Verify no empty mappings
    assert len(mappings) > 0


@pytest.mark.parametrize("minimal_mapping", [True, False])
def test_model_mappings_contain_expected_models(minimal_mapping: bool):
    """Test that model mappings contain expected OpenAI models."""
    if minimal_mapping:
        mapping = DEFAULT_MINIMAL_MODEL_MAPPINGS
        manager = OllamaManager(DEFAULT_MINIMAL_MODEL_MAPPINGS)
    else:
        mapping = DEFAULT_MODEL_MAPPINGS
        manager = OllamaManager()

    expected_openai_models = mapping.keys()

    for model in expected_openai_models:
        assert model in manager.model_mappings


@pytest.mark.parametrize("minimal_mapping", [True, False])
def test_model_mappings_contain_expected_ollama_models(minimal_mapping: bool):
    """Test that model mappings contain expected Ollama models."""
    if minimal_mapping:
        mapping = DEFAULT_MINIMAL_MODEL_MAPPINGS
        manager = OllamaManager(DEFAULT_MINIMAL_MODEL_MAPPINGS)
    else:
        mapping = DEFAULT_MODEL_MAPPINGS
        manager = OllamaManager()

    expected_ollama_models = mapping.values

    ollama_models = list(manager.model_mappings.values())
    for model in expected_ollama_models:
        assert model in ollama_models
