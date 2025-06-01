"""
Tests for OpenAI client routing with module state management.
Minimal mocking approach - only mock network responses and environment variables.
"""
import pytest
import os
import json
import tempfile
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# -- Ours --
import reflex_llms

# -- Tests --
from tests.conftest import *

# =======================================
#                 Cleanup
# =======================================


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
def temp_config_file():
    """Create a temporary reflex.json configuration file."""
    config_data = {
        "openai_base_url": "https://custom-api.example.com/v1",
        "azure_api_version": "2024-06-01",
        "azure_base_url": "https://custom-azure.openai.azure.com",
        "preference_order": ["reflex", "azure", "openai"],
        "timeout": 10.0,
        "reflex_server": {
            "host": "127.0.0.1",
            "port": 8080,
            "image": "ollama/ollama:latest",
            "container_name": "test-reflex-server",
            "auto_setup": True,
            "model_mappings": {
                "minimal_setup": True,
                "model_mapping": {
                    "gpt-3.5-turbo": "llama3.2:3b",
                    "gpt-4": "llama3.1:8b"
                },
                "minimal_model_mapping": {
                    "gpt-3.5-turbo": "llama3.2:3b",
                    "text-embedding-ada-002": "nomic-embed-text"
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    yield temp_path, config_data

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with reflex.json for file discovery testing."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "reflex.json"

    config_data = {
        "openai_base_url": "https://discovered-api.example.com/v1",
        "preference_order": ["openai", "reflex"],
        "timeout": 8.0,
        "reflex_server": {
            "port": 9999,
            "model_mappings": {
                "minimal_setup": False
            }
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    yield temp_dir, config_data

    # Cleanup
    shutil.rmtree(temp_dir)
