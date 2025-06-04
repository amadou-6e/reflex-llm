from tests.test_models import *


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
def test_setup_model_mapping_without_container(model_manager_no_container, capsys):
    """Test setup_model_mapping when Ollama is not running."""
    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        model_manager_no_container.setup_model_mapping()

    captured = capsys.readouterr()
    assert "Setting up OpenAI-compatible models" in captured.out


def test_setup_model_mapping_empty_mappings(model_manager):
    """Test setup with empty model mappings."""
    # Temporarily empty the mappings
    original_mappings = model_manager.model_mappings.copy()
    model_manager.model_mappings = {}

    try:
        result = model_manager.setup_model_mapping()
        assert result is True  # Success with 0/0 models
    finally:
        # Restore original mappings
        model_manager.model_mappings = original_mappings
