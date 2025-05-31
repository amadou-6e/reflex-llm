"""
Tests for the ContainerHandler class using real Docker.
"""
import pytest
import time
import os
import uuid
import docker
from pathlib import Path

# -- Ours --
from reflex_llms.containers import ContainerHandler
# -- Tests --
from tests.conftest import *
from tests.utils import nuke_dir, clear_port

# =======================================
#                 Cleanup
# =======================================


@pytest.fixture(autouse=True)
def cleanup_temp_dir():
    """Clean up temp files using OS-agnostic commands."""
    nuke_dir(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    yield
    nuke_dir(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_containers():
    """
    Automatically clean up containers whose names start with 'test-ollama'
    at the end of the module.
    """
    yield  # let tests run

    client = docker.from_env()
    for container in client.containers.list(all=True):  # include stopped
        name = container.name
        if name.startswith("test-ollama") or name.startswith("integration-test-ollama"):
            print(f"Cleaning up container: {name}")
            try:
                container.stop(timeout=5)
            except docker.errors.APIError:
                pass  # maybe already stopped
            try:
                container.remove(force=True)
            except docker.errors.APIError as e:
                print(f"Failed to remove container {name}: {e}")


@pytest.fixture
def clear_port_11435():
    clear_port(11435, "test-ollama")


@pytest.fixture
def clear_port_11436():
    clear_port(11436, "integration-test-ollama")


@pytest.fixture
def clear_port_19999():
    clear_port(19999, "test-ollama")


# =======================================
#              Directories
# =======================================


@pytest.fixture
def temp_dir() -> Path:
    """Create a unique temporary directory for test isolation."""
    run_id = str(uuid.uuid4())[:8]
    temp_path = Path(TEMP_DIR, "test_containers", run_id)
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


# =======================================
#              Handlers
# =======================================


@pytest.fixture
def container_handler(temp_dir: Path):
    """Create ContainerHandler instance with temporary path."""
    name = f"test-ollama-{uuid.uuid4().hex[:8]}"
    return ContainerHandler(
        host="127.0.0.1",
        port=11435,
        image="ollama/ollama:latest",
        container_name=name,
        data_path=temp_dir,
    )


# =======================================
#              Container Management
# =======================================


@pytest.fixture
def remove_test_container():
    """Helper to remove test containers by name pattern."""

    def _remove_container(container_name: str):
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            container.remove(force=True)
            print(f"Removed existing container: {container_name}")
        except docker.errors.NotFound:
            # Container doesn't exist, that's fine
            pass
        except Exception as e:
            print(f"Warning: Failed to remove container {container_name}: {str(e)}")

    return _remove_container
