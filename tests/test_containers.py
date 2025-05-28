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


@pytest.fixture
def nested_temp_dir(tmp_path) -> Path:
    """Create nested temporary directories for testing."""
    nested_path = Path(tmp_path, "deep", "nested", "ollama")
    return nested_path


@pytest.fixture
def existing_data_dir(tmp_path) -> Path:
    """Create an existing data directory with test content."""
    data_path = Path(tmp_path, "existing_ollama")
    data_path.mkdir(parents=True, exist_ok=True)

    # Create a test file in the directory
    test_file = Path(data_path, "test.txt")
    test_file.write_text("test content")

    return data_path


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


@pytest.fixture
def integration_container_handler(temp_dir: Path):
    """Create ContainerHandler instance for integration tests."""
    name = f"integration-test-ollama-{uuid.uuid4().hex[:8]}"
    return ContainerHandler(
        host="127.0.0.1",
        port=11436,
        container_name=name,
        startup_timeout=None,
        data_path=temp_dir,
    )


@pytest.fixture
def custom_port_handler():
    """Create ContainerHandler with custom port for testing."""
    name = f"test-ollama-{uuid.uuid4().hex[:8]}"
    return ContainerHandler(
        host="192.168.1.100",
        port=9090,
        container_name=name,
    )


@pytest.fixture
def high_port_handler():
    """Create ContainerHandler with high port number for port testing."""
    name = f"test-ollama-{uuid.uuid4().hex[:8]}"
    return ContainerHandler(
        port=19999,
        container_name=name,
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


# =======================================
#              Basic Tests
# =======================================


def test_init_creates_data_directory(temp_dir: Path):
    """Test that initialization creates data directory."""
    handler = ContainerHandler(data_path=temp_dir)

    assert temp_dir.exists()
    assert handler.data_path == temp_dir


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


def test_is_port_open_when_nothing_running(high_port_handler: ContainerHandler):
    """Test port checking when nothing is running on the port."""
    assert high_port_handler._is_port_open() is False


def test_get_api_url(container_handler: ContainerHandler):
    """Test getting API URL."""
    url = container_handler.api_url
    assert url == "http://127.0.0.1:11435"


def test_get_openai_compatible_url(container_handler: ContainerHandler):
    """Test getting OpenAI-compatible API URL."""
    url = container_handler.openai_compatible_url
    assert url == "http://127.0.0.1:11435/v1"


def test_get_api_url_custom_host_port(custom_port_handler: ContainerHandler):
    """Test getting API URL with custom host and port."""
    url = custom_port_handler.api_url
    assert url == "http://192.168.1.100:9090"
    url_v1 = custom_port_handler.openai_compatible_url
    assert url_v1 == "http://192.168.1.100:9090/v1"


def test_data_path_creation_with_nested_dirs(nested_temp_dir: Path):
    """Test that nested data directories are created properly."""
    handler = ContainerHandler(data_path=nested_temp_dir)

    assert nested_temp_dir.exists()
    assert nested_temp_dir.is_dir()


def test_container_handler_initialization_with_existing_data_dir(existing_data_dir: Path):
    """Test initialization when data directory already exists."""
    handler = ContainerHandler(data_path=existing_data_dir)

    # Test file should be preserved
    test_file = Path(existing_data_dir, "test.txt")

    # Directory should still exist and file should be preserved
    assert existing_data_dir.exists()
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_configuration_validation():
    """Test that configuration is properly validated."""
    name = f"test-ollama-{uuid.uuid4().hex[:8]}"
    handler = ContainerHandler(
        host="custom-host",
        port=9999,
        image="custom/ollama:tag",
        container_name=name,
    )

    assert handler.host == "custom-host"
    assert handler.port == 9999
    assert handler.image == "custom/ollama:tag"
    assert handler.container_name == name
    assert handler.api_url == "http://custom-host:9999"
    assert handler.openai_compatible_url == "http://custom-host:9999/v1"


# =======================================
#           Container Operations
# =======================================


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


# =======================================
#          Integration Tests
# =======================================


@pytest.mark.usefixtures("clear_port_11436")
def test_ensure_running_creates_new_container(integration_container_handler: ContainerHandler,
                                              remove_test_container):
    """Test ensure_running when creating a new container."""
    if not integration_container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Clean up any existing container
        remove_test_container(integration_container_handler.container_name)

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
        while time.time() - start_time < 60 * 3:  # 3 minute timeout
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


@pytest.mark.usefixtures("clear_port_11435")
def test_ensure_running_with_existing_stopped_container(container_handler: ContainerHandler,
                                                        remove_test_container):
    """Test ensure_running when container exists but is stopped."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create a stopped container
        remove_test_container(container_handler.container_name)

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


# =======================================
#            Error Handling
# =======================================


def test_docker_not_available_error_handling():
    """Test error handling when Docker is not available."""
    # Create handler with a container name that won't conflict
    name = f"no-docker-test-{uuid.uuid4().hex[:8]}"
    handler = ContainerHandler(container_name=name)

    if handler._is_docker_running():
        pytest.skip("Docker is running, cannot test Docker unavailable scenario")

    # When Docker is not running, ensure_running should raise RuntimeError
    with pytest.raises(RuntimeError, match="Docker is not running"):
        handler.ensure_running()


# =======================================
#         Ollama-Specific Tests
# =======================================


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
