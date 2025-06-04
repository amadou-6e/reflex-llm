import pytest

# -- Ours --
from reflex_llms.settings import DEFAULT_MODEL_MAPPINGS
from reflex_llms.models import OllamaManager
# -- Tests --
from tests.test_models import *
from tests.conftest import *


@pytest.mark.slow
@pytest.mark.integration
def test_all_default_models_accessible_after_setup(model_manager_full: OllamaManager):
    """
    Test that all models in model_mappings are accessible after setup_model_mapping().
    
    This is a comprehensive integration test that:
    1. Runs the full setup_model_mapping() process
    2. Verifies every OpenAI model name from the mappings is accessible
    3. Confirms the models can be listed and exist in Ollama
    
    This test is slow as it downloads multiple models and should be marked accordingly.
    """
    # Use a subset of models for faster testing

    print(f"Testing setup of {len(DEFAULT_MODEL_MAPPINGS)} model mappings...")

    # Run the setup process
    setup_success = model_manager_full.setup_model_mapping()

    assert setup_success, "Model setup failed - models may not be available for download"

    # Verify setup was successful
    assert setup_success, "setup_model_mapping() should return True for successful setup"

    # Get the current list of available models
    available_models = model_manager_full.list_models()
    available_model_names = [model["name"] for model in available_models]

    print(f"Available models after setup: {available_model_names}")

    # Verify all OpenAI model names are now accessible
    missing_models = []
    for openai_name, ollama_name in DEFAULT_MODEL_MAPPINGS.items():
        print(f"Checking accessibility of: {openai_name}")

        # Check if the OpenAI-named model exists
        if not model_manager_full.model_exists(openai_name):
            missing_models.append(openai_name)
            print(f"FAIL: {openai_name} is NOT accessible")
        else:
            print(f"PASS: {openai_name} is accessible")

        # Also verify the underlying Ollama model exists
        if not model_manager_full.model_exists(ollama_name):
            missing_models.append(f"{ollama_name} (underlying)")
            print(f"FAIL: {ollama_name} (underlying model) is NOT accessible")
        else:
            print(f"PASS: {ollama_name} (underlying model) is accessible")

    # Assert all models are accessible
    assert len(missing_models) == 0, f"Models not accessible after setup: {missing_models}"

    # Verify models appear in the model list
    for openai_name in DEFAULT_MODEL_MAPPINGS.keys():
        assert any(openai_name in model_name for model_name in available_model_names), \
            f"OpenAI model {openai_name} not found in available models list"

    print(f"SUCCESS: All {len(DEFAULT_MODEL_MAPPINGS)} models successfully set up and accessible")


@pytest.mark.slow
@pytest.mark.integration
def test_model_mapping_setup_resilience(model_manager_full: OllamaManager):
    """
    Test setup behavior when some models are not available for download.
    
    This tests the resilience of the setup process when some models
    cannot be downloaded or are not available in the registry.
    """

    # Setup should handle failures gracefully
    setup_success = model_manager_full.setup_model_mapping()

    # Should return False since not all models succeeded
    assert setup_success is False, "setup_model_mapping() should return False when some models fail"

    # Valid model should still be accessible
    assert model_manager_full.model_exists(
        "gpt-valid"), "Valid model should be accessible even when others fail"

    # Invalid model should not be accessible
    assert not model_manager_full.model_exists(
        "gpt-invalid"), "Invalid model should not be accessible"

    print("SUCCESS: Setup gracefully handled missing models")
