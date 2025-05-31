import pytest
import time
import uuid
import docker
from pathlib import Path
from contextlib import contextmanager

# -- Ours --
from reflex_llms.containers import ContainerHandler
# -- Tests --
from tests.test_containers import *
from tests.conftest import *


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


# =======================================
#              Container State Fixtures
# =======================================


@pytest.fixture
def ensure_no_container():
    """Ensure no container exists before and after test."""

    @contextmanager
    def _ensure_clean(container_handler: ContainerHandler):
        # Clean up before test
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        try:
            yield
        finally:
            # Clean up after test
            cleanup = container_handler._get_container()
            if cleanup:
                cleanup.remove(force=True)

    return _ensure_clean


@pytest.fixture
def stopped_container():
    """Create a stopped container for testing."""

    @contextmanager
    def _create_stopped(container_handler: ContainerHandler):
        # Ensure clean state first
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create stopped container
        container_handler._pull_image()
        container = container_handler._create_container()

        # Verify it's stopped
        container.reload()
        assert container.status in ["created", "exited"]

        try:
            yield container
        finally:
            # Cleanup
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Already removed

    return _create_stopped


@pytest.fixture
def running_container():
    """Create a running container for testing."""

    @contextmanager
    def _create_running(container_handler: ContainerHandler):
        # Ensure clean state first
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create and start container
        container_handler._pull_image()
        container = container_handler._create_container()
        container_handler._start_container()

        # Verify it's running
        container.reload()
        assert container.status == "running"

        try:
            yield container
        finally:
            # Cleanup
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Already removed

    return _create_running


@pytest.fixture
def ready_container():
    """Create a running container that's ready to accept requests."""

    @contextmanager
    def _create_ready(container_handler: ContainerHandler, timeout: int = 120):
        # Ensure clean state first
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create, start and wait for ready
        container_handler._pull_image()
        container = container_handler._create_container()
        container_handler._start_container()
        container_handler._wait_for_ready(timeout=timeout)

        # Verify it's ready
        assert container_handler._is_port_open()

        try:
            yield container
        finally:
            # Cleanup
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Already removed

    return _create_ready


@pytest.fixture
def conflicting_container():
    """Create a container on the same port with different name for conflict testing."""

    @contextmanager
    def _create_conflicting(container_handler: ContainerHandler, suffix: str = "conflict"):
        conflicting_name = f"{container_handler.container_name}-{suffix}"
        conflicting_handler = ContainerHandler(host=container_handler.host,
                                               port=container_handler.port,
                                               container_name=conflicting_name,
                                               data_path=container_handler.data_path)

        # Ensure clean state
        existing = conflicting_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create and start conflicting container
        conflicting_handler._pull_image()
        container = conflicting_handler._create_container()
        conflicting_handler._start_container()

        try:
            yield conflicting_handler, container
        finally:
            # Cleanup
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass

    return _create_conflicting


@pytest.fixture
def clean_container_state():
    """Ensure clean container state before test and cleanup after."""

    @contextmanager
    def _clean_state(container_handler: ContainerHandler):
        # Clean up before test
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Clear the port as well
        clear_port(container_handler.port, container_handler.container_name.split('-')[0])

        try:
            yield
        finally:
            # Clean up after test
            cleanup = container_handler._get_container()
            if cleanup:
                cleanup.remove(force=True)

    return _clean_state


@pytest.fixture
def container_lifecycle():
    """Manage complete container lifecycle for a test."""
    containers_to_cleanup = []

    def _register_container(container_handler: ContainerHandler):
        containers_to_cleanup.append(container_handler)
        return container_handler

    yield _register_container

    # Cleanup all registered containers
    for handler in containers_to_cleanup:
        container = handler._get_container()
        if container:
            try:
                container.remove(force=True)
                print(f"Cleaned up container: {handler.container_name}")
            except Exception as e:
                print(f"Failed to cleanup container {handler.container_name}: {e}")


# =======================================
#              Direct Use Fixtures
# =======================================


@pytest.fixture
def ensure_stopped_container():
    """Create a stopped container that's automatically cleaned up."""
    container_ref = None

    def create_stopped(container_handler: ContainerHandler):
        nonlocal container_ref

        # Clean up any existing container
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create stopped container
        container_handler._pull_image()
        container_ref = container_handler._create_container()

        # Verify it's stopped
        container_ref.reload()
        assert container_ref.status in ["created", "exited"]

        return container_ref

    yield create_stopped

    # Cleanup
    if container_ref:
        try:
            container_ref.remove(force=True)
        except docker.errors.NotFound:
            pass


@pytest.fixture
def ensure_running_container():
    """Create a running container that's automatically cleaned up."""
    container_ref = None

    def create_running(container_handler: ContainerHandler):
        nonlocal container_ref

        # Clean up any existing container
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create and start container
        container_handler._pull_image()
        container_ref = container_handler._create_container()
        container_handler._start_container()

        # Verify it's running
        container_ref.reload()
        assert container_ref.status == "running"

        return container_ref

    yield create_running

    # Cleanup
    if container_ref:
        try:
            container_ref.remove(force=True)
        except docker.errors.NotFound:
            pass


@pytest.fixture
def ensure_ready_container():
    """Create a ready container that's automatically cleaned up."""
    container_ref = None

    def create_ready(container_handler: ContainerHandler, timeout: int = 120):
        nonlocal container_ref

        # Clean up any existing container
        existing = container_handler._get_container()
        if existing:
            existing.remove(force=True)

        # Create, start and wait for ready
        container_handler._pull_image()
        container_ref = container_handler._create_container()
        container_handler._start_container()
        container_handler._wait_for_ready(timeout=timeout)

        # Verify it's ready
        assert container_handler._is_port_open()

        return container_ref

    yield create_ready

    # Cleanup
    if container_ref:
        try:
            container_ref.remove(force=True)
        except docker.errors.NotFound:
            pass


# =======================================
#              Tests
# =======================================


@pytest.mark.usefixtures("clear_port_11436")
def test_start_creates_new_container(
    integration_container_handler: ContainerHandler,
    clean_container_state,
):
    """Test start when creating a new container."""
    if not integration_container_handler._is_docker_running():
        pytest.skip("Docker not running")

    with clean_container_state(integration_container_handler):
        # This should create and start a new container
        integration_container_handler.start()

        # Verify container exists and is running
        container = integration_container_handler._get_container()
        assert container is not None
        container.reload()
        assert container.status == "running"

        # Verify Ollama becomes ready (this might take a while)
        start_time = time.time()
        while time.time() - start_time < 60 * 3:  # 3 minute timeout
            if integration_container_handler._is_port_open():
                break
            time.sleep(2)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_with_existing_stopped_container(
    container_handler: ContainerHandler,
    ensure_stopped_container,
):
    """Test start when container exists but is stopped."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    container = ensure_stopped_container(container_handler)

    # Now start should start it
    container_handler.start()

    # Verify it's running
    container.reload()
    assert container.status == "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_with_running_container(
    container_handler: ContainerHandler,
    ensure_running_container,
):
    """Test start with exists_ok=False when container is already running."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    ensure_running_container(container_handler)

    # start with exists_ok=False should raise error
    with pytest.raises(RuntimeError, match="is already running"):
        container_handler.start(exists_ok=False)

    # Container should still be running after the error
    assert container_handler._is_container_running()


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_with_force(
    container_handler: ContainerHandler,
    ensure_running_container,
):
    """Test start with exists_ok=False and force=True deletes and recreates."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    original_container = ensure_running_container(container_handler)
    original_id = original_container.id

    # start with exists_ok=False and force=True should delete and recreate
    container_handler.start(exists_ok=False, force=True)

    # Should have a new container (different ID)
    new_container = container_handler._get_container()
    assert new_container is not None
    assert new_container.id != original_id
    assert new_container.status == "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_restart_flag(container_handler: ContainerHandler, ensure_running_container):
    """Test start with restart=True restarts existing container."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    container = ensure_running_container(container_handler)

    # Record the start time
    container.reload()
    original_start_time = container.attrs['State']['StartedAt']

    # Wait a moment to ensure timestamp difference
    time.sleep(1)

    # start with restart=True should restart the container
    container_handler.start(restart=True)

    # Verify container was restarted (new start time)
    container.reload()
    new_start_time = container.attrs['State']['StartedAt']
    assert new_start_time != original_start_time
    assert container.status == "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_attach_port_false_with_running_service(
    container_handler: ContainerHandler,
    ensure_ready_container,
    container_lifecycle,
):
    """Test start with attach_port=False when service is already running on port."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Register for cleanup
    container_lifecycle(container_handler)

    ensure_ready_container(container_handler, timeout=60)

    # Create a different handler with same port but different name
    different_handler = ContainerHandler(
        host=container_handler.host,
        port=container_handler.port,
        container_name=f"{container_handler.container_name}-different")

    container_lifecycle(different_handler)

    # With attach_port=False, should not attach to existing service
    # but since force=False, should raise error about port being in use
    with pytest.raises(RuntimeError, match="Port.*in use"):
        different_handler.start(attach_port=False, force=False)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_attach_port_false_with_force(
    container_handler: ContainerHandler,
    ensure_running_container,
    container_lifecycle,
):
    """Test start with attach_port=False and force=True clears port."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Register containers for cleanup
    container_lifecycle(container_handler)

    original_container = ensure_running_container(container_handler)

    # Create a different handler with same port
    different_handler = ContainerHandler(
        host=container_handler.host,
        port=container_handler.port,
        container_name=f"{container_handler.container_name}-different")

    container_lifecycle(different_handler)

    # With attach_port=False and force=True, should clear port and create new container
    different_handler.start(attach_port=False, force=True)

    # Should have created a new container
    new_container = different_handler._get_container()
    assert new_container is not None
    assert new_container.status == "running"

    # Original container should be stopped
    original_container.reload()
    assert original_container.status != "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_with_stopped_container(
    container_handler: ContainerHandler,
    ensure_stopped_container,
):
    """Test start with exists_ok=False when container exists but is stopped."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    ensure_stopped_container(container_handler)

    # start with exists_ok=False should raise error even for stopped containers
    with pytest.raises(RuntimeError, match="Container.*exists"):
        container_handler.start(exists_ok=False)

    # Container should still exist
    assert container_handler._get_container() is not None


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_force_with_stopped_container(
    container_handler: ContainerHandler,
    ensure_stopped_container,
):
    """Test start with exists_ok=False and force=True deletes stopped container."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    original_container = ensure_stopped_container(container_handler)
    original_id = original_container.id

    # start with exists_ok=False and force=True should delete and recreate
    container_handler.start(exists_ok=False, force=True)

    # Should have a new container
    new_container = container_handler._get_container()
    assert new_container is not None
    assert new_container.id != original_id
    assert new_container.status == "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_attach_port_true_uses_existing_service(
    container_handler: ContainerHandler,
    ensure_ready_container,
    container_lifecycle,
):
    """Test start with attach_port=True (default) uses existing service."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Register for cleanup
    container_lifecycle(container_handler)

    original_container = ensure_ready_container(container_handler, timeout=60)
    original_id = original_container.id

    # Create a different handler with same port
    different_handler = ContainerHandler(
        host=container_handler.host,
        port=container_handler.port,
        container_name=f"{container_handler.container_name}-different")

    container_lifecycle(different_handler)

    # With attach_port=True (default), should use existing service
    different_handler.start(attach_port=True)

    # Should not have created a new container
    new_container = different_handler._get_container()
    assert new_container is None  # No container with the different name

    # Original container should still be running
    original_container.reload()
    assert original_container.status == "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_force_clears_port_when_in_use(
    container_handler: ContainerHandler,
    conflicting_container,
    container_lifecycle,
):
    """Test start with force=True clears port when it's in use."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    # Register main container for cleanup
    container_lifecycle(container_handler)

    with conflicting_container(container_handler,
                               suffix="conflict") as (conflicting_handler, conflict_container):
        # Register conflicting container for cleanup
        container_lifecycle(conflicting_handler)

        # Now try to start our main container with force=True
        container_handler.start(force=True, attach_port=False)

        # Our container should be running
        our_container = container_handler._get_container()
        assert our_container is not None
        assert our_container.status == "running"

        # Conflicting container should be stopped
        conflict_container.reload()
        assert conflict_container.status != "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_with_all_flags_combination(
    container_handler: ContainerHandler,
    ensure_running_container,
):
    """Test start with combination of flags: exists_ok=False, force=True, restart=True."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    original_container = ensure_running_container(container_handler)
    original_id = original_container.id

    # start with multiple flags should force delete and recreate
    container_handler.start(exists_ok=False, force=True, restart=True, attach_port=False)

    # Should have a new container (force=True overrides exists_ok=False)
    new_container = container_handler._get_container()
    assert new_container is not None
    assert new_container.id != original_id
    assert new_container.status == "running"


@pytest.mark.usefixtures("clear_port_11435")
def test_start_restart_with_stopped_container(
    container_handler: ContainerHandler,
    ensure_stopped_container,
):
    """Test start with restart=True when container is stopped should just start it."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    container = ensure_stopped_container(container_handler)
    container_id = container.id

    # start with restart=True should just start the stopped container
    container_handler.start(restart=True)

    # Should be the same container, now running
    container.reload()
    assert container.id == container_id
    assert container.status == "running"
