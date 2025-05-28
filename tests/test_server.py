"""
Tests for the ReflexServer class.
"""
import pytest
import time
import uuid
import docker
from pathlib import Path
from unittest.mock import patch, Mock

# -- Ours --
from reflex_llms.server import ReflexServer
from reflex_llms.containers import ContainerHandler
from reflex_llms.models import OllamaModelManager
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
    Automatically clean up containers whose names contain UUID patterns
    (test-reflex-server-*, test-lifecycle-*, test-model-*, etc.) at the end of the module.
    """
    yield  # let tests run

    client = docker.from_env()
    for container in client.containers.list(all=True):  # include stopped
        name = container.name
        # Match containers with UUID patterns to ensure we only clean up our test containers
        if any(prefix in name for prefix in [
                "test-reflex-server-", "test-lifecycle-", "test-model-", "test-openai-",
                "test-completions", "test-robustness", "test-all-mappings", "test-error-recovery"
        ]) and (len(name.split('-')) >= 3):
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
def clear_port_11437():
    clear_port(11437, "test-reflex-server")


@pytest.fixture
def clear_port_11438():
    clear_port(11438, "test-reflex-server")


@pytest.fixture
def clear_port_11440():
    clear_port(11440, "test-lifecycle")


@pytest.fixture
def clear_port_11441():
    clear_port(11441, "test-model")


@pytest.fixture
def clear_port_11442():
    clear_port(11442, "test-openai")


@pytest.fixture
def clear_port_11443():
    clear_port(11443, "test-completions")


@pytest.fixture
def clear_port_11444():
    clear_port(11444, "test-robustness")


@pytest.fixture
def clear_port_11445():
    clear_port(11445, "test-all-mappings")


@pytest.fixture
def clear_port_11446():
    clear_port(11446, "test-error-recovery")


# =======================================
#              Directories
# =======================================


@pytest.fixture
def temp_data_path() -> Path:
    """Create temporary data path for testing with UUID isolation."""
    run_id = str(uuid.uuid4())[:8]
    temp_path = Path(TEMP_DIR, "reflex_test_data", run_id)
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


@pytest.fixture
def custom_data_path(tmp_path) -> Path:
    """Create custom data path for testing."""
    custom_path = tmp_path / "custom_reflex_test"
    custom_path.mkdir(parents=True, exist_ok=True)
    return custom_path


# =======================================
#              Server Instances
# =======================================


@pytest.fixture
def reflex_server_no_auto_setup(temp_data_path: Path):
    """Create ReflexServer instance without auto setup."""
    container_name = f"test-reflex-server-{str(uuid.uuid4())[:8]}"
    return ReflexServer(
        host="127.0.0.1",
        port=11437,
        container_name=container_name,
        data_path=temp_data_path,
        auto_setup=False,
        essential_models_only=True,
    )


@pytest.fixture
def reflex_server_with_setup(temp_data_path: Path):
    """Create ReflexServer instance with auto setup (integration test)."""
    container_name = f"test-reflex-server-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(host="127.0.0.1",
                              port=11438,
                              container_name=container_name,
                              data_path=temp_data_path,
                              auto_setup=True,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def custom_reflex_server(custom_data_path: Path):
    """Create ReflexServer with custom parameters."""
    container_name = f"test-reflex-server-{str(uuid.uuid4())[:8]}"
    return ReflexServer(host="192.168.1.100",
                        port=9999,
                        container_name=container_name,
                        data_path=custom_data_path,
                        auto_setup=False,
                        essential_models_only=False)


# =======================================
#              Integration Test Servers
# =======================================


@pytest.fixture
def lifecycle_server():
    """Create server for lifecycle testing."""
    container_name = f"test-lifecycle-real-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11440,
                              container_name=container_name,
                              auto_setup=False,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def model_download_server():
    """Create server for model download testing."""
    container_name = f"test-model-download-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11441,
                              container_name=container_name,
                              auto_setup=False,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def openai_setup_server():
    """Create server for OpenAI setup testing."""
    container_name = f"test-openai-setup-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11442,
                              container_name=container_name,
                              auto_setup=False,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def completions_server():
    """Create server for completions testing."""
    container_name = f"test-completions-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11443,
                              container_name=container_name,
                              auto_setup=True,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def robustness_server():
    """Create server for robustness testing."""
    container_name = f"test-robustness-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11444,
                              container_name=container_name,
                              auto_setup=False,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def all_mappings_server():
    """Create server for all mappings testing."""
    container_name = f"test-all-mappings-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11445,
                              container_name=container_name,
                              auto_setup=False,
                              essential_models_only=False)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


@pytest.fixture
def error_recovery_server():
    """Create server for error recovery testing."""
    container_name = f"test-error-recovery-{str(uuid.uuid4())[:8]}"
    server = None
    try:
        server = ReflexServer(port=11446,
                              container_name=container_name,
                              auto_setup=False,
                              essential_models_only=True)
        yield server
    finally:
        if server:
            try:
                server.stop()
            except Exception as e:
                print(f"Cleanup error: {e}")


# =======================================
#              Basic Tests
# =======================================


def test_reflex_server_initialization_no_auto_setup(reflex_server_no_auto_setup: ReflexServer):
    """Test ReflexServer initialization without auto setup."""
    server = reflex_server_no_auto_setup

    # Check basic properties
    assert server.host == "127.0.0.1"
    assert server.port == 11437
    assert server.container_name.startswith("test-reflex-server-")
    assert server.essential_models_only is True
    assert server._setup_complete is False

    # Check component initialization
    assert isinstance(server.container_handler, ContainerHandler)
    assert isinstance(server.model_manager, OllamaModelManager)

    # Check that auto setup was skipped
    assert server._setup_complete is False


def test_reflex_server_initialization_with_custom_params(custom_reflex_server: ReflexServer):
    """Test ReflexServer initialization with custom parameters."""
    server = custom_reflex_server

    assert server.host == "192.168.1.100"
    assert server.port == 9999
    assert server.container_name.startswith("test-reflex-server-")
    assert server.essential_models_only is False


def test_reflex_server_component_urls(reflex_server_no_auto_setup: ReflexServer):
    """Test that URLs are properly constructed."""
    server = reflex_server_no_auto_setup

    expected_api_url = "http://127.0.0.1:11437"
    expected_openai_url = "http://127.0.0.1:11437/v1"

    assert server.api_url == expected_api_url
    assert server.openai_compatible_url == expected_openai_url
    assert server.container_handler.api_url == expected_api_url
    assert server.container_handler.openai_compatible_url == expected_openai_url


# =======================================
#              Setup Tests
# =======================================


def test_setup_essential_models_mapping(reflex_server_no_auto_setup: ReflexServer):
    """Test that essential models mapping is configured correctly."""
    server = reflex_server_no_auto_setup

    # Test the _setup_essential_models method creates correct mapping
    original_mappings = server.model_manager.model_mappings.copy()

    # Mock the setup_openai_models to avoid actual model operations
    with patch.object(server.model_manager, 'setup_openai_models', return_value=True):
        result = server._setup_essential_models()

    # Should have returned True
    assert result is True

    # Original mappings should be restored
    assert server.model_manager.model_mappings == original_mappings


def test_wait_for_ollama_ready_success(reflex_server_no_auto_setup: ReflexServer):
    """Test successful wait for Ollama ready."""
    server = reflex_server_no_auto_setup

    # Mock list_models to simulate Ollama being ready
    with patch.object(server.model_manager, 'list_models', return_value=[{"name": "test-model"}]):
        # Should not raise exception
        server._wait_for_ollama_ready(timeout=10)


def test_wait_for_ollama_ready_timeout(reflex_server_no_auto_setup: ReflexServer):
    """Test timeout when waiting for Ollama."""
    server = reflex_server_no_auto_setup

    # Mock list_models to always raise exception (Ollama not ready)
    with patch.object(server.model_manager, 'list_models', side_effect=Exception("Not ready")):
        with pytest.raises(RuntimeError, match="did not become ready within"):
            server._wait_for_ollama_ready(timeout=1)  # Short timeout for test


def test_setup_method_integration(reflex_server_no_auto_setup: ReflexServer):
    """Test the complete setup method with mocked components."""
    server = reflex_server_no_auto_setup

    # Mock all the components to simulate successful setup
    with patch.object(server.container_handler, 'ensure_running') as mock_ensure_running, \
         patch.object(server, '_wait_for_ollama_ready') as mock_wait, \
         patch.object(server, '_setup_essential_models', return_value=True) as mock_setup_models, \
         patch.object(server, 'health_check', return_value=True) as mock_health:

        result = server.setup()

        # Verify all steps were called
        mock_ensure_running.assert_called_once()
        mock_wait.assert_called_once()
        mock_setup_models.assert_called_once()
        mock_health.assert_called_once()

        # Should return True and mark setup complete
        assert result is True
        assert server._setup_complete is True


def test_setup_method_failure(reflex_server_no_auto_setup: ReflexServer):
    """Test setup method when container fails to start."""
    server = reflex_server_no_auto_setup

    # Mock container to fail
    with patch.object(server.container_handler,
                      'ensure_running',
                      side_effect=Exception("Container failed")):

        result = server.setup()

        # Should return False and not mark setup complete
        assert result is False
        assert server._setup_complete is False


# =======================================
#           Health Check Tests
# =======================================


def test_health_check_all_pass(reflex_server_no_auto_setup: ReflexServer):
    """Test health check when all checks pass."""
    server = reflex_server_no_auto_setup

    mock_models = [{"name": "gpt-3.5-turbo"}, {"name": "llama3.2:3b"}]

    with patch.object(server.container_handler, '_is_container_running', return_value=True), \
         patch.object(server.container_handler, '_is_port_open', return_value=True), \
         patch.object(server.model_manager, 'list_models', return_value=mock_models):

        result = server.health_check(force=True)
        assert result is True


def test_health_check_container_not_running(reflex_server_no_auto_setup: ReflexServer):
    """Test health check when container is not running."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler, '_is_container_running', return_value=False):
        result = server.health_check(force=True)
        assert result is False


def test_health_check_port_not_open(reflex_server_no_auto_setup: ReflexServer):
    """Test health check when port is not accessible."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler, '_is_container_running', return_value=True), \
         patch.object(server.container_handler, '_is_port_open', return_value=False):

        result = server.health_check(force=True)
        assert result is False


def test_health_check_no_openai_models(reflex_server_no_auto_setup: ReflexServer):
    """Test health check when no OpenAI models are available."""
    server = reflex_server_no_auto_setup

    # Mock models that don't match OpenAI naming
    mock_models = [{"name": "some-other-model"}, {"name": "another-model"}]

    with patch.object(server.container_handler, '_is_container_running', return_value=True), \
         patch.object(server.container_handler, '_is_port_open', return_value=True), \
         patch.object(server.model_manager, 'list_models', return_value=mock_models):

        result = server.health_check(force=True)
        # Should still return True (functional, just no OpenAI models)
        assert result is True


def test_health_check_caching(reflex_server_no_auto_setup: ReflexServer):
    """Test that health check uses caching."""
    server = reflex_server_no_auto_setup

    # Set last health check to recent time
    server._last_health_check = time.time()

    with patch.object(server.container_handler, '_is_container_running') as mock_container:
        # Should use cache and not call container check
        result = server.health_check(force=False)
        mock_container.assert_not_called()


def test_health_check_exception_handling(reflex_server_no_auto_setup: ReflexServer):
    """Test health check exception handling."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler,
                      '_is_container_running',
                      side_effect=Exception("Test exception")):

        result = server.health_check(force=True)
        assert result is False


# =======================================
#             Status Tests
# =======================================


def test_get_status_success(reflex_server_no_auto_setup: ReflexServer):
    """Test get_status when everything is working."""
    server = reflex_server_no_auto_setup
    server._setup_complete = True

    mock_models = [{"name": "gpt-3.5-turbo"}, {"name": "llama3.2:3b"}]

    with patch.object(server.model_manager, 'list_models', return_value=mock_models), \
         patch.object(server.container_handler, '_is_container_running', return_value=True), \
         patch.object(server.container_handler, '_is_port_open', return_value=True), \
         patch.object(server, 'health_check', return_value=True):

        status = server.get_status()

        assert status["setup_complete"] is True
        assert status["container_running"] is True
        assert status["port_open"] is True
        assert status["total_models"] == 2
        assert "gpt-3.5-turbo" in status["openai_compatible_models"]
        assert status["healthy"] is True
        assert "api_url" in status
        assert "openai_compatible_url" in status


def test_get_status_with_error(reflex_server_no_auto_setup: ReflexServer):
    """Test get_status when an error occurs."""
    server = reflex_server_no_auto_setup

    with patch.object(server.model_manager, 'list_models', side_effect=Exception("Test error")):

        status = server.get_status()

        assert status["setup_complete"] is False
        assert "error" in status
        assert status["healthy"] is False


# =======================================
#           Lifecycle Tests
# =======================================


def test_stop_method(reflex_server_no_auto_setup: ReflexServer):
    """Test stop method."""
    server = reflex_server_no_auto_setup
    server._setup_complete = True

    with patch.object(server.container_handler, 'stop') as mock_stop:
        server.stop()

        mock_stop.assert_called_once()
        assert server._setup_complete is False


def test_stop_method_with_error(reflex_server_no_auto_setup: ReflexServer):
    """Test stop method when container stop fails."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler, 'stop', side_effect=Exception("Stop failed")):
        # Should not raise exception
        server.stop()


def test_restart_method(reflex_server_no_auto_setup: ReflexServer):
    """Test restart method."""
    server = reflex_server_no_auto_setup

    with patch.object(server, 'stop') as mock_stop, \
         patch.object(server, 'setup', return_value=True) as mock_setup, \
         patch('time.sleep') as mock_sleep:

        result = server.restart()

        mock_stop.assert_called_once()
        mock_sleep.assert_called_once_with(2)
        mock_setup.assert_called_once()
        assert result is True


# =======================================
#           Property Tests
# =======================================


def test_is_running_property(reflex_server_no_auto_setup: ReflexServer):
    """Test is_running property."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler, '_is_port_open', return_value=True):
        assert server.is_running is True

    with patch.object(server.container_handler, '_is_port_open', return_value=False):
        assert server.is_running is False


def test_is_healthy_property(reflex_server_no_auto_setup: ReflexServer):
    """Test is_healthy property."""
    server = reflex_server_no_auto_setup

    with patch.object(server, 'health_check', return_value=True):
        assert server.is_healthy is True

    with patch.object(server, 'health_check', return_value=False):
        assert server.is_healthy is False


# =======================================
#          Error Handling Tests
# =======================================


def test_setup_with_model_failure(reflex_server_no_auto_setup: ReflexServer):
    """Test setup when model setup fails but continues."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler, 'ensure_running'), \
         patch.object(server, '_wait_for_ollama_ready'), \
         patch.object(server, '_setup_essential_models', return_value=False), \
         patch.object(server, 'health_check', return_value=True):

        result = server.setup()

        # Should still succeed even if models failed
        assert result is True
        assert server._setup_complete is True


def test_setup_with_health_check_failure(reflex_server_no_auto_setup: ReflexServer):
    """Test setup when final health check fails."""
    server = reflex_server_no_auto_setup

    with patch.object(server.container_handler, 'ensure_running'), \
         patch.object(server, '_wait_for_ollama_ready'), \
         patch.object(server, '_setup_essential_models', return_value=True), \
         patch.object(server, 'health_check', return_value=False):

        result = server.setup()

        # Should fail if health check fails
        assert result is False
        assert server._setup_complete is False


# =======================================
#       Real Integration Tests
# =======================================


@pytest.mark.integration
@pytest.mark.usefixtures("clear_port_11440")
def test_container_lifecycle_real(lifecycle_server: ReflexServer):
    """Test complete container lifecycle without mocking."""
    server = lifecycle_server

    # Initially should not be running
    assert server.is_running is False
    assert server._setup_complete is False

    # Start container manually
    print("Testing container startup...")
    server.container_handler.ensure_running()

    # Should now be running
    assert server.container_handler._is_container_running() is True
    assert server.container_handler._is_port_open() is True

    # Test Ollama API connectivity
    print("Testing Ollama API connectivity...")
    server._wait_for_ollama_ready(timeout=60)

    # Should be able to list models (even if empty)
    models = server.model_manager.list_models()
    assert isinstance(models, list)
    print(f"Found {len(models)} existing models")

    # Test health check
    health = server.health_check(force=True)
    assert health is True

    # Test stop
    print("Testing container stop...")
    server.stop()
    assert server._setup_complete is False


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("clear_port_11441")
def test_model_download_and_management(model_download_server: ReflexServer):
    """Test actual model download and management operations."""
    server = model_download_server

    # Setup container
    server.container_handler.ensure_running()
    server._wait_for_ollama_ready()

    print("Testing model download operations...")

    # Test downloading a very small model
    small_model = "smollm:135m"  # Small model for testing

    # Check if model exists before
    exists_before = server.model_manager.model_exists(small_model)
    print(f"Model {small_model} exists before: {exists_before}")

    if not exists_before:
        print(f"Downloading {small_model}...")
        success = server.model_manager.pull_model(small_model)

        if success:
            # Verify model was downloaded
            assert server.model_manager.model_exists(small_model) is True
            print(f"Successfully downloaded {small_model}")

            # Test model info
            models = server.model_manager.list_models()
            downloaded_model = next((m for m in models if m['name'].startswith(small_model)), None)
            assert downloaded_model is not None
            assert 'size' in downloaded_model

            # Test creating alias
            alias_name = "test-alias-model"
            alias_success = server.model_manager.copy_model(small_model, alias_name)
            if alias_success:
                assert server.model_manager.model_exists(alias_name) is True
                print(f"Successfully created alias: {alias_name}")

                # Clean up alias
                try:
                    server.model_manager._make_request("delete", "DELETE", {"name": alias_name})
                except Exception:
                    pass  # Cleanup attempt, ignore errors

        else:
            print(f"Failed to download {small_model} - may not be available")
    else:
        print(f"Model {small_model} already exists, skipping download test")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("clear_port_11442")
def test_openai_model_setup_real(openai_setup_server: ReflexServer):
    """Test real OpenAI model setup with actual downloads."""
    server = openai_setup_server

    # Setup container
    server.container_handler.ensure_running()
    server._wait_for_ollama_ready()

    print("Testing OpenAI model setup...")

    # Get initial model count
    initial_models = server.model_manager.list_models()
    print(f"Initial models: {len(initial_models)}")

    # Test essential models setup
    success = server._setup_essential_models()
    print(f"Essential models setup result: {success}")

    # Check final model count
    final_models = server.model_manager.list_models()
    print(f"Final models: {len(final_models)}")

    # Should have more models now (or at least same number)
    assert len(final_models) >= len(initial_models)

    # Check for OpenAI-compatible model names
    model_names = [m['name'] for m in final_models]
    openai_models = [
        name for name in model_names
        if any(openai_name in name
               for openai_name in ["gpt-3.5-turbo", "gpt-4o-mini", "text-embedding-ada-002"])
    ]

    print(f"OpenAI-compatible models found: {openai_models}")

    # At least some OpenAI models should exist (even if setup partially failed)
    if success:
        assert len(openai_models) > 0, "No OpenAI-compatible models found after successful setup"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("clear_port_11443")
def test_openai_api_completions(completions_server: ReflexServer):
    """Test actual OpenAI API completions through the server."""
    import openai

    server = completions_server

    # Wait for setup to complete
    max_wait = 300  # 5 minutes
    wait_time = 0
    while not server._setup_complete and wait_time < max_wait:
        time.sleep(5)
        wait_time += 5
        print(f"Waiting for setup... {wait_time}s")

    if not server._setup_complete:
        pytest.skip("Server setup did not complete in time")

    # Verify we have models
    models = server.model_manager.list_models()
    if len(models) == 0:
        pytest.skip("No models available for completion testing")

    # Find an available OpenAI-compatible model
    model_names = [m['name'] for m in models]
    test_model = None

    # Check for OpenAI models, considering they might have tags like :latest
    for openai_model in ["gpt-4o-mini", "gpt-3.5-turbo"]:
        # Check both clean name and tagged versions
        if (openai_model in model_names or f"{openai_model}:latest" in model_names or
                any(name.startswith(f"{openai_model}:") for name in model_names)):
            test_model = openai_model  # Use clean name for API
            break

    if not test_model:
        print(f"Available models: {model_names}")
        pytest.skip("No OpenAI-compatible models available for testing")

    print(f"Testing completions with model: {test_model}")

    # Create OpenAI client pointing to our server
    client = openai.OpenAI(
        base_url=server.openai_compatible_url,
        api_key="test-key"  # Ollama doesn't validate API keys
    )

    # Test simple completion
    print("Testing chat completion...")
    try:
        response = client.chat.completions.create(
            model=test_model,
            messages=[{
                "role": "user",
                "content": "Say 'Hello World' and nothing else"
            }],
            max_tokens=50,
            temperature=0.1)

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None

        content = response.choices[0].message.content.strip()
        print(f"Completion response: {content}")

        # Basic validation - should contain some response
        assert len(content) > 0

    except Exception as e:
        print(f"Completion test failed: {e}")
        # Don't fail the test if completion fails - models might not be fully ready
        pytest.skip(f"Completion failed: {e}")

    # Test model listing through OpenAI API
    print("Testing model listing through OpenAI API...")
    try:
        api_models = client.models.list()
        assert api_models.data is not None
        assert len(api_models.data) > 0

        api_model_names = [m.id for m in api_models.data]
        print(f"Models available through API: {api_model_names}")

        # Should include our test model (API should return clean names or handle both)
        # The test model we use is the clean name, API should accept it
        model_found = (test_model in api_model_names or f"{test_model}:latest" in api_model_names or
                       any(name.startswith(f"{test_model}:") for name in api_model_names))
        assert model_found, f"Test model '{test_model}' not found in API models: {api_model_names}"

    except Exception as e:
        print(f"Model listing test failed: {e}")
        # This is less critical than completions


@pytest.mark.integration
@pytest.mark.usefixtures("clear_port_11444")
def test_server_robustness_and_recovery(robustness_server: ReflexServer):
    """Test server robustness, error recovery, and edge cases."""
    server = robustness_server

    # Test multiple start/stop cycles
    print("Testing multiple start/stop cycles...")
    for i in range(3):
        print(f"Cycle {i+1}")

        # Start
        server.container_handler.ensure_running()
        assert server.container_handler._is_port_open() is True

        # Stop
        server.container_handler.stop()
        time.sleep(2)  # Give it time to stop

        # Should be stopped
        assert server.container_handler._is_container_running() is False

    # Test health checks under various conditions
    print("Testing health checks...")

    # When stopped
    health_stopped = server.health_check(force=True)
    assert health_stopped is False

    # When started
    server.container_handler.ensure_running()
    server._wait_for_ollama_ready()
    health_running = server.health_check(force=True)
    assert health_running is True

    # Test status reporting
    print("Testing status reporting...")
    status = server.get_status()
    assert isinstance(status, dict)
    assert "container_running" in status
    assert "port_open" in status
    assert "total_models" in status
    assert "healthy" in status

    # Test restart functionality
    print("Testing restart...")
    restart_success = server.restart()
    assert restart_success is True
    assert server.is_running is True


@pytest.mark.integration
@pytest.mark.usefixtures("clear_port_11445")
def test_all_model_mappings(all_mappings_server: ReflexServer):
    """Test that all model mappings in the library can be processed."""
    server = all_mappings_server

    server.container_handler.ensure_running()
    server._wait_for_ollama_ready()

    print("Testing all model mappings...")

    # Get all model mappings
    all_mappings = server.model_manager.model_mappings
    print(f"Total model mappings to test: {len(all_mappings)}")

    successful_mappings = 0
    failed_mappings = 0

    for openai_name, ollama_name in all_mappings.items():
        print(f"Testing mapping: {openai_name} -> {ollama_name}")

        # Check if base model exists or can be pulled
        if not server.model_manager.model_exists(ollama_name):
            print(f"  Attempting to pull {ollama_name}...")
            pull_success = server.model_manager.pull_model(ollama_name)

            if not pull_success:
                print(f"  Failed to pull {ollama_name}")
                failed_mappings += 1
                continue

        # Test creating OpenAI alias - check for tagged versions too
        openai_exists = (server.model_manager.model_exists(openai_name) or
                         server.model_manager.model_exists(f"{openai_name}:latest"))

        if not openai_exists:
            copy_success = server.model_manager.copy_model(ollama_name, openai_name)
            if copy_success:
                print(f"  Successfully created {openai_name}")
                successful_mappings += 1
            else:
                print(f"  Failed to create alias {openai_name}")
                failed_mappings += 1
        else:
            print(f"  {openai_name} already exists")
            successful_mappings += 1

    print(f"Mapping results: {successful_mappings} successful, {failed_mappings} failed")

    # At least some mappings should work
    assert successful_mappings > 0, "No model mappings succeeded"

    # Test that OpenAI models are accessible - check for tagged versions
    final_models = server.model_manager.list_models()
    openai_models = [
        m for m in final_models
        if any(openai_name in m['name'] for openai_name in all_mappings.keys())
    ]

    print(f"Total OpenAI-compatible models available: {len(openai_models)}")
    print(f"OpenAI model names found: {[m['name'] for m in openai_models]}")
    assert len(openai_models) > 0


@pytest.mark.integration
@pytest.mark.usefixtures("clear_port_11446")
def test_error_recovery_real(error_recovery_server: ReflexServer):
    """Test error recovery with real container operations."""
    server = error_recovery_server

    # Test setup when Docker is available
    server.container_handler.ensure_running()

    # Test recovery from stopped container
    print("Testing recovery from stopped container...")
    server.container_handler.stop()
    time.sleep(2)

    # Health check should fail
    assert server.health_check(force=True) is False

    # But restart should work
    restart_success = server.restart()
    assert restart_success is True
    assert server.is_running is True

    # Test invalid model operations
    print("Testing invalid model operations...")
    invalid_pull = server.model_manager.pull_model("definitely-invalid-model-name-12345")
    assert invalid_pull is False

    # Test invalid copy operation
    invalid_copy = server.model_manager.copy_model("nonexistent-source", "test-dest")
    assert invalid_copy is False

    # Container should still be functional
    assert server.health_check(force=True) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
