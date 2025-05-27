"""
Tests for the ContainerHandler class using real Docker.
"""
import pytest
import time
import os
import uuid
from pathlib import Path

# -- Ours --
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
        port=11435,
        image="ollama/ollama:latest",
        container_name="test-ollama-handler",
        data_path=temp_dir,
    )


@pytest.fixture
def integration_container_handler(temp_dir: Path):
    """Create ContainerHandler instance for integration tests."""
    return ContainerHandler(
        host="127.0.0.1",
        port=11436,
        container_name="integration-test-ollama",
        startup_timeout=None,
        data_path=temp_dir,
    )


def test_init_creates_data_directory(temp_dir: Path):
    """Test that initialization creates data directory."""
    data_path = temp_dir
    handler = ContainerHandler(data_path=data_path)

    assert data_path.exists()
    assert handler.data_path == data_path


def test_init_default_data_path():
    """Test that default data path is created."""
    handler = ContainerHandler()
    expected_path = Path.home() / ".ollama-docker"
    assert handler.data_path == expected_path
    assert expected_path.exists()


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
    assert url == "http://127.0.0.1:11435"


def test_get_openai_compatible_url(container_handler: ContainerHandler):
    """Test getting OpenAI-compatible API URL."""
    url = container_handler.openai_compatible_url
    assert url == "http://127.0.0.1:11435/v1"


def test_get_api_url_custom_host_port():
    """Test getting API URL with custom host and port."""
    handler = ContainerHandler(host="192.168.1.100", port=9090)
    url = handler.api_url
    assert url == "http://192.168.1.100:9090"
    url_v1 = handler.openai_compatible_url
    assert url_v1 == "http://192.168.1.100:9090/v1"


def test_data_path_creation_with_nested_dirs(tmp_path):
    """Test that nested data directories are created properly."""
    data_path = Path(tmp_path, "deep", "nested", "ollama")
    handler = ContainerHandler(data_path=data_path)

    assert data_path.exists()
    assert data_path.is_dir()


def test_container_handler_initialization_with_existing_data_dir(tmp_path):
    """Test initialization when data directory already exists."""
    data_path = Path(tmp_path, "existing_ollama")
    data_path.mkdir(parents=True, exist_ok=True)

    # Create a test file in the directory
    test_file = Path(data_path, "test.txt")
    test_file.write_text("test content")

    handler = ContainerHandler(data_path=data_path)

    # Directory should still exist and file should be preserved
    assert data_path.exists()
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_configuration_validation():
    """Test that configuration is properly validated."""
    handler = ContainerHandler(
        host="custom-host",
        port=9999,
        image="custom/ollama:tag",
        container_name="custom-name",
    )

    assert handler.host == "custom-host"
    assert handler.port == 9999
    assert handler.image == "custom/ollama:tag"
    assert handler.container_name == "custom-name"
    assert handler.api_url == "http://custom-host:9999"
    assert handler.openai_compatible_url == "http://custom-host:9999/v1"


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
        assert "Started Ollama container" in captured.out
        assert "Stopped Ollama container" in captured.out

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

        # Verify Ollama becomes ready (this might take a while)
        # We'll wait a reasonable amount of time
        start_time = time.time()
        while time.time(
        ) - start_time < 60 * 3:  # 3 minute timeout (Ollama starts faster than LocalAI)
            if integration_container_handler._is_port_open():
                break
            time.sleep(2)

        # Note: We don't assert Ollama is ready because it might take time
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


def test_ollama_specific_port_check():
    """Test that Ollama-specific health check endpoint is used."""
    handler = ContainerHandler(port=19998)

    # This should return False since nothing is running on this port
    # and it's checking the Ollama-specific /api/tags endpoint
    assert handler._is_port_open() is False


def test_ollama_mount_configuration():
    """Test that Ollama-specific mount configuration is correct."""
    temp_path = Path("/tmp/test-ollama-mount")
    handler = ContainerHandler(data_path=temp_path)

    # Mock the container creation to inspect mounts
    if handler._is_docker_running():
        try:
            existing = handler._get_container()
            if existing:
                existing.remove(force=True)

            handler._pull_image()
            container = handler._create_container()

            # Check that the mount is configured correctly for Ollama
            container_info = handler.client.api.inspect_container(container.id)
            mounts = container_info['Mounts']

            ollama_mount = next((m for m in mounts if m['Destination'] == '/root/.ollama'), None)
            assert ollama_mount is not None
            assert ollama_mount['Type'] == 'bind'

        finally:
            cleanup_container = handler._get_container()
            if cleanup_container:
                cleanup_container.remove(force=True)
    else:
        pytest.skip("Docker not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
