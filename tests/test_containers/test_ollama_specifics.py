import pytest
import uuid

# -- Ours --
from reflex_llms.containers import ContainerHandler
# -- Tests --
from tests.test_containers import *
from tests.conftest import *


def test_ollama_specific_port_check():
    """Test that Ollama-specific health check endpoint is used."""
    name = f"test-ollama-{uuid.uuid4().hex[:8]}"
    handler = ContainerHandler(port=19998, container_name=name)

    # This should return False since nothing is running on this port
    # and it's checking the Ollama-specific /api/tags endpoint
    assert handler._is_port_open() is False


def test_ollama_mount_configuration(container_handler: ContainerHandler, remove_test_container):
    """Test that Ollama-specific mount configuration is correct."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        remove_test_container(container_handler.container_name)

        container_handler._pull_image()
        container = container_handler._create_container()

        # Check that the mount is configured correctly for Ollama
        container_info = container_handler.client.api.inspect_container(container.id)
        mounts = container_info['Mounts']

        ollama_mount = next((m for m in mounts if m['Destination'] == '/root/.ollama'), None)
        assert ollama_mount is not None
        assert ollama_mount['Type'] == 'bind'

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)
