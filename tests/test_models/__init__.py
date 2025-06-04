"""
Tests for the OllamaModelManager class.
"""
import pytest
import time
import docker
import uuid
from pathlib import Path

# -- Ours --
from reflex_llms.models import OllamaManager
from reflex_llms.containers import ContainerHandler

# -- Tests --
from tests.utils import nuke_dir, clear_port
from tests.conftest import *


def cleanup_containers():
    client = docker.from_env()
    for container in client.containers.list(all=True):
        name = container.name
        if name.startswith("test-ollama"):
            print(f"Cleaning up container: {name}")
            try:
                container.stop(timeout=5)
            except docker.errors.APIError:
                pass
            try:
                container.remove(force=True)
            except docker.errors.APIError as e:
                print(f"Failed to remove container {name}: {e}")


@pytest.fixture(autouse=True, scope="module")
def cleanup_temp_dir():
    """Clean up temp files using OS-agnostic commands."""
    nuke_dir(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    yield
    nuke_dir(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def temp_dir() -> Path:
    """Create a unique temporary directory for test isolation."""
    run_id = str(uuid.uuid4())[:8]
    temp_path = Path(TEMP_DIR, "test_containers", run_id)
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_containers():
    """
    Automatically clean up containers whose names start with 'test-ollama'
    at the end of the module.
    """
    cleanup_containers()
    yield
    cleanup_containers()


@pytest.fixture(scope="session", autouse=True)
def run_uuid():
    """Generate a unique identifier for test runs."""
    return str(uuid.uuid4())[:8]


@pytest.fixture(scope="module")
def ollama_container(temp_dir, run_uuid):
    """Start Ollama container for testing session."""

    container_handler = ContainerHandler(
        host="127.0.0.1",
        port=11435,
        container_name=f"test-ollama-manager-{run_uuid}",
        data_path=Path(temp_dir, "test-ollama"),
    )

    try:
        # Ensure Ollama container is running
        container_handler.start()

        # Wait a bit for Ollama to be fully ready
        time.sleep(2)

        yield container_handler

    finally:
        # Clean up after tests
        try:
            container_handler.stop()
        except Exception as e:
            print(f"Error stopping container: {e}")


@pytest.fixture(scope="module")
def ollama_container_models(temp_dir, run_uuid):
    container_handler = ContainerHandler(
        host="127.0.0.1",
        port=11437,
        container_name=f"test-ollama-model-manager-{run_uuid}",
        data_path=Path(temp_dir, "test-ollama-models"),
    )

    try:
        container_handler.start()
        time.sleep(2)

        yield container_handler

    finally:
        try:
            container_handler.stop()
        except Exception as e:
            print(f"Error stopping container: {e}")


@pytest.fixture
def model_manager(ollama_container):
    """Create OllamaModelManager instance using the test container."""
    return OllamaManager(ollama_url=ollama_container.api_url)


@pytest.fixture
def model_manager_full(ollama_container_models):
    """Create OllamaModelManager instance using the test container."""
    return OllamaManager(ollama_url=ollama_container_models.api_url)


@pytest.fixture
def model_manager_no_container():
    """Create OllamaModelManager instance without container for error testing."""
    return OllamaManager(ollama_url="http://127.0.0.1:65432")
