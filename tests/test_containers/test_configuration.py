import uuid
from pathlib import Path

# -- Ours --
from reflex_llms.containers import ContainerHandler
# -- Tests --
from tests.test_containers import *
from tests.conftest import *

# =======================================
#              Fixtures
# =======================================


@pytest.fixture
def high_port_handler():
    """Create ContainerHandler with high port number for port testing."""
    name = f"test-ollama-{uuid.uuid4().hex[:8]}"
    return ContainerHandler(
        port=19999,
        container_name=name,
    )


@pytest.fixture
def nested_temp_dir() -> Path:
    """Create nested temporary directories for testing."""
    nested_path = Path(TEMP_DIR, "deep", "nested", "ollama")
    return nested_path


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
def existing_data_dir() -> Path:
    """Create an existing data directory with test content."""
    data_path = Path(TEMP_DIR, "existing_ollama")
    data_path.mkdir(parents=True, exist_ok=True)

    # Create a test file in the directory
    test_file = Path(data_path, "test.txt")
    test_file.write_text("test content")

    return data_path


# =======================================
#              Tests
# =======================================


def test_init_creates_data_directory():
    """Test that initialization creates data directory."""
    handler = ContainerHandler(data_path=TEMP_DIR)

    assert TEMP_DIR.exists()
    assert handler.data_path == TEMP_DIR


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
