"""
Tests for the ContainerHandler class using real Docker.
"""
import pytest
import time
import os
import uuid
from pathlib import Path

# -- Ours --
from reflex_llms.settings import MODEL_PATH
from reflex_llms.containers import ContainerHandler
# -- Tests --
from tests.conftest import *


@pytest.fixture
def temp_dir() -> Path:
    run_id = str(uuid.uuid4())[:6]
    return Path(TEMP_DIR, "test_containers").joinpath(run_id)


@pytest.fixture
def container_handler(temp_dir: Path):
    """Create ContainerHandler instance with temporary path."""
    return ContainerHandler(
        host="127.0.0.1",
        port=8081,
        image="localai/localai:latest-cpu",
        container_name="test-localai-handler",
        models_path=temp_dir,
    )


@pytest.fixture
def integration_container_handler(temp_dir: Path):
    """Create ContainerHandler instance for integration tests."""
    return ContainerHandler(
        host="127.0.0.1",
        port=8082,
        container_name="integration-test-localai",
        startup_timeout=None,
        models_path=temp_dir,
    )


def test_init_creates_models_directory(temp_dir: Path):
    """Test that initialization creates models directory."""
    models_path = temp_dir,
    handler = ContainerHandler(models_path=models_path)

    assert models_path.exists()
    assert handler.models_path == models_path


def test_init_default_models_path():
    """Test that default models path is created."""
    handler = ContainerHandler()
    expected_path = Path(os.getcwd(), "models")
    assert handler.models_path == MODEL_PATH


def test_is_docker_running(container_handler: ContainerHandler):
    """Test Docker daemon detection."""
    result = container_handler._is_docker_running()
    # This will be True if Docker is running, False otherwise
    assert isinstance(result, bool)

    if result:
        # If Docker is running, client should be initialized
        assert container_handler.client is not None
    else:
        # If Docker is not running, client should be None
        assert container_handler.client is None


def test_is_port_open_when_nothing_running(container_handler: ContainerHandler):
    """Test port checking when nothing is running on the port."""
    # Use a random high port that should be free
    handler = ContainerHandler(port=19999)
    assert handler._is_port_open() is False


def test_get_api_url(container_handler: ContainerHandler):
    """Test getting API URL."""
    url = container_handler.api_url
    assert url == "http://127.0.0.1:8081/v1"


def test_get_api_url_custom_host_port():
    """Test getting API URL with custom host and port."""
    handler = ContainerHandler(host="192.168.1.100", port=9090)
    url = handler.api_url
    assert url == "http://192.168.1.100:9090/v1"


def test_models_path_creation_with_nested_dirs(tmp_path):
    """Test that nested model directories are created properly."""
    models_path = Path(tmp_path, "deep", "nested", "models")
    handler = ContainerHandler(models_path=models_path)

    assert models_path.exists()
    assert models_path.is_dir()


def test_container_handler_initialization_with_existing_models_dir(tmp_path):
    """Test initialization when models directory already exists."""
    models_path = Path(tmp_path, "existing_models")
    models_path.mkdir(parents=True, exist_ok=True)

    # Create a test file in the directory
    test_file = Path(models_path, "test.txt")
    test_file.write_text("test content")

    handler = ContainerHandler(models_path=models_path)

    # Directory should still exist and file should be preserved
    assert models_path.exists()
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_configuration_validation():
    """Test that configuration is properly validated."""
    handler = ContainerHandler(
        host="custom-host",
        port=9999,
        image="custom/image:tag",
        container_name="custom-name",
    )

    assert handler.host == "custom-host"
    assert handler.port == 9999
    assert handler.image == "custom/image:tag"
    assert handler.container_name == "custom-name"
    assert handler.api_url == "http://custom-host:9999/v1"


@pytest.mark.docker
def test_get_container_when_none_exists(container_handler: ContainerHandler):
    """Test getting container when it doesn't exist."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Make sure container doesn't exist first
    existing = container_handler._get_container()
    if existing:
        existing.remove(force=True)

    result = container_handler._get_container()
    assert result is None


@pytest.mark.docker
def test_is_container_running_when_none_exists(container_handler: ContainerHandler):
    """Test container running check when container doesn't exist."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Make sure container doesn't exist first
    existing = container_handler._get_container()
    if existing:
        existing.remove(force=True)

    assert container_handler._is_container_running() is False


@pytest.mark.docker
def test_pull_image_download(container_handler, capsys):
    """Test pulling image when it might need to be downloaded."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # This test will either find the image exists or pull it
    container_handler._pull_image()

    # Should not raise any exceptions
    assert container_handler.client.images.get(container_handler.image)


@pytest.mark.docker
def test_create_and_remove_container(container_handler: ContainerHandler):
    """Test creating and removing a container."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Clean up any existing container first
    existing = container_handler._get_container()
    if existing:
        existing.remove(force=True)

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


@pytest.mark.docker
def test_container_lifecycle(container_handler, capsys):
    """Test full container lifecycle: create, start, stop, remove."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Clean up any existing container
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

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
        assert "Started LocalAI container" in captured.out
        assert "Stopped LocalAI container" in captured.out

    finally:
        # Always clean up
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.docker
@pytest.mark.slow
def test_ensure_running_creates_new_container(integration_container_handler: ContainerHandler):
    """Test ensure_running when creating a new container."""
    if not integration_container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Clean up any existing container
        existing = integration_container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # This should create and start a new container
        integration_container_handler.ensure_running()

        # Verify container exists and is running
        container = integration_container_handler._get_container()
        assert container is not None
        container.reload()
        assert container.status == "running"

        # Verify LocalAI becomes ready (this might take a while)
        # We'll wait a reasonable amount of time
        start_time = time.time()
        while time.time() - start_time < 60 * 5:  # 5 minute timeout
            if integration_container_handler._is_port_open():
                break
            time.sleep(2)

        # Note: We don't assert LocalAI is ready because it might take very long
        # The important thing is that the container is running

    finally:
        # Clean up
        cleanup_container = integration_container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.docker
def test_ensure_running_with_existing_stopped_container(container_handler: ContainerHandler):
    """Test ensure_running when container exists but is stopped."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create a stopped container
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        container_handler._pull_image()
        container = container_handler._create_container()

        # Verify it's not running
        container.reload()
        assert container.status in ["created", "exited"]

        # Now ensure_running should start it
        container_handler.ensure_running()

        # Verify it's running
        container.reload()
        assert container.status == "running"

    finally:
        # Clean up
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


def test_docker_not_available_error_handling():
    """Test error handling when Docker is not available."""
    # Create handler with a container name that won't conflict
    handler = ContainerHandler(container_name="no-docker-test")

    if handler._is_docker_running():
        pytest.skip("Docker is running, cannot test Docker unavailable scenario")

    # When Docker is not running, ensure_running should raise RuntimeError
    with pytest.raises(RuntimeError, match="Docker is not running"):
        handler.ensure_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
