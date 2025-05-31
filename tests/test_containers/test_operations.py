import pytest

# -- Ours --
from reflex_llms.containers import ContainerHandler
# -- Tests --
from tests.test_containers import *
from tests.conftest import *


def test_get_container_when_none_exists(container_handler: ContainerHandler, remove_test_container):
    """Test getting container when it doesn't exist."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Make sure container doesn't exist first
    remove_test_container(container_handler.container_name)

    result = container_handler._get_container()
    assert result is None


def test_is_container_running_when_none_exists(container_handler: ContainerHandler,
                                               remove_test_container):
    """Test container running check when container doesn't exist."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Make sure container doesn't exist first
    remove_test_container(container_handler.container_name)

    assert container_handler._is_container_running() is False


def test_pull_image_download(container_handler: ContainerHandler, capsys):
    """Test pulling image when it might need to be downloaded."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # This test will either find the image exists or pull it
    container_handler._pull_image()

    # Should not raise any exceptions
    assert container_handler.client.images.get(container_handler.image)


def test_create_and_remove_container(container_handler: ContainerHandler, remove_test_container):
    """Test creating and removing a container."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Clean up any existing container first
    remove_test_container(container_handler.container_name)

    # Pull image first
    container_handler._pull_image()

    # Create container
    container = container_handler._create_container()
    assert container is not None

    # Verify container exists
    found_container = container_handler._get_container()
    assert found_container is not None
    assert found_container.name == container_handler.container_name

    # Clean up
    found_container.remove(force=True)

    # Verify it's gone
    assert container_handler._get_container() is None


@pytest.mark.usefixtures("clear_port_11435")
def test_container_lifecycle(container_handler: ContainerHandler, remove_test_container, capsys):
    """Test full container lifecycle: create, start, stop, remove."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Clean up any existing container
        remove_test_container(container_handler.container_name)

        # Pull image
        container_handler._pull_image()

        # Create container
        container = container_handler._create_container()
        assert container is not None

        # Start container
        container_handler._start_container()

        # Check if it's running
        container.reload()
        assert container.status == "running"

        # Stop container
        container_handler.stop()

        # Verify output
        captured = capsys.readouterr()
        assert "Started Ollama container" in captured.out
        assert "Stopped Ollama container" in captured.out

    finally:
        # Always clean up
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)
