# -- Ours --
from reflex_llms.models import OllamaManager


# Initialization Tests
def test_default_initialization():
    """Test default initialization values."""
    manager = OllamaManager()
    assert manager.ollama_url == "http://127.0.0.1:11434"
    assert isinstance(manager.model_mappings, dict)
    assert len(manager.model_mappings) > 0


def test_custom_url_initialization():
    """Test initialization with custom URL."""
    custom_url = "http://localhost:8080"
    manager = OllamaManager(ollama_url=custom_url)
    assert manager.ollama_url == custom_url


def test_model_mappings_content():
    """Test that model mappings contain expected OpenAI models."""
    manager = OllamaManager()
    expected_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "text-embedding-ada-002"]

    for model in expected_models:
        assert model in manager.model_mappings
        assert isinstance(manager.model_mappings[model], str)
        assert len(manager.model_mappings[model]) > 0
