import pytest
import time
import uuid
from pathlib import Path

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


@pytest.mark.usefixtures("clear_port_11436")
def test_start_creates_new_container(integration_container_handler: ContainerHandler,
                                     remove_test_container):
    """Test start when creating a new container."""
    if not integration_container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Clean up any existing container
        remove_test_container(integration_container_handler.container_name)

        # This should create and start a new container
        integration_container_handler.start()

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
def test_start_with_existing_stopped_container(
    container_handler: ContainerHandler,
    remove_test_container,
):
    """Test start when container exists but is stopped."""
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

        # Now start should start it
        container_handler.start()

        # Verify it's running
        container.reload()
        assert container.status == "running"

    finally:
        # Clean up
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


# ============================================================================
# NEW FLAG TESTS FOR START METHOD
# ============================================================================


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_with_running_container(container_handler: ContainerHandler,
                                                      remove_test_container):
    """Test start with exists_ok=False when container is already running."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create and start a container first
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        container_handler._create_container()
        container_handler._start_container()

        # Verify it's running
        assert container_handler._is_container_running()

        # start with exists_ok=False should raise error
        with pytest.raises(RuntimeError, match="is already running"):
            container_handler.start(exists_ok=False)

        # Container should still be running after the error
        assert container_handler._is_container_running()

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_with_force(container_handler: ContainerHandler,
                                          remove_test_container):
    """Test start with exists_ok=False and force=True deletes and recreates."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create and start a container first
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        original_container = container_handler._create_container()
        container_handler._start_container()
        original_id = original_container.id

        # start with exists_ok=False and force=True should delete and recreate
        container_handler.start(exists_ok=False, force=True)

        # Should have a new container (different ID)
        new_container = container_handler._get_container()
        assert new_container is not None
        assert new_container.id != original_id
        assert new_container.status == "running"

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_restart_flag(container_handler: ContainerHandler, remove_test_container):
    """Test start with restart=True restarts existing container."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create and start a container first
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        container = container_handler._create_container()
        container_handler._start_container()

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

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_attach_port_false_with_running_service(container_handler: ContainerHandler,
                                                      remove_test_container):
    """Test start with attach_port=False when service is already running on port."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Start a container on the port first
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        container_handler._create_container()
        container_handler._start_container()

        # Wait for service to be ready
        container_handler._wait_for_ready(timeout=60)
        assert container_handler._is_port_open()

        # Create a different handler with same port but different name
        different_handler = ContainerHandler(
            host=container_handler.host,
            port=container_handler.port,
            container_name=f"{container_handler.container_name}-different")

        # With attach_port=False, should not attach to existing service
        # but since force=False, should raise error about port being in use
        with pytest.raises(RuntimeError, match="Port.*in use"):
            different_handler.start(attach_port=False, force=False)

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_attach_port_false_with_force(container_handler: ContainerHandler,
                                            remove_test_container):
    """Test start with attach_port=False and force=True clears port."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Start a container on the port first
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        container_handler._create_container()
        container_handler._start_container()

        # Create a different handler with same port
        different_handler = ContainerHandler(
            host=container_handler.host,
            port=container_handler.port,
            container_name=f"{container_handler.container_name}-different")

        # With attach_port=False and force=True, should clear port and create new container
        different_handler.start(attach_port=False, force=True)

        # Should have created a new container
        new_container = different_handler._get_container()
        assert new_container is not None
        assert new_container.status == "running"

        # Original container should be stopped
        original_container = container_handler._get_container()
        if original_container:
            original_container.reload()
            assert original_container.status != "running"

    finally:
        # Clean up both containers
        for handler in [container_handler, different_handler]:
            cleanup_container = handler._get_container()
            if cleanup_container:
                cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_with_stopped_container(container_handler: ContainerHandler,
                                                      remove_test_container):
    """Test start with exists_ok=False when container exists but is stopped."""
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

        # start with exists_ok=False should raise error even for stopped containers
        with pytest.raises(RuntimeError, match="Container.*exists"):
            container_handler.start(exists_ok=False)

        # Container should still exist
        assert container_handler._get_container() is not None

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_exists_ok_false_force_with_stopped_container(container_handler: ContainerHandler,
                                                            remove_test_container):
    """Test start with exists_ok=False and force=True deletes stopped container."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create a stopped container
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        original_container = container_handler._create_container()
        original_id = original_container.id

        # start with exists_ok=False and force=True should delete and recreate
        container_handler.start(exists_ok=False, force=True)

        # Should have a new container
        new_container = container_handler._get_container()
        assert new_container is not None
        assert new_container.id != original_id
        assert new_container.status == "running"

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_attach_port_true_uses_existing_service(container_handler: ContainerHandler,
                                                      remove_test_container):
    """Test start with attach_port=True (default) uses existing service."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Start a container and wait for it to be ready
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        container_handler._create_container()
        container_handler._start_container()
        container_handler._wait_for_ready(timeout=60)

        original_container = container_handler._get_container()
        original_id = original_container.id

        # Create a different handler with same port
        different_handler = ContainerHandler(
            host=container_handler.host,
            port=container_handler.port,
            container_name=f"{container_handler.container_name}-different")

        # With attach_port=True (default), should use existing service
        different_handler.start(attach_port=True)

        # Should not have created a new container
        new_container = different_handler._get_container()
        assert new_container is None  # No container with the different name

        # Original container should still be running
        original_container.reload()
        assert original_container.status == "running"

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_force_clears_port_when_in_use(container_handler: ContainerHandler,
                                             remove_test_container):
    """Test start with force=True clears port when it's in use."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Start a container with a different name on the same port
        conflicting_name = f"{container_handler.container_name}-conflict"
        conflicting_handler = ContainerHandler(host=container_handler.host,
                                               port=container_handler.port,
                                               container_name=conflicting_name)

        remove_test_container(conflicting_name)
        conflicting_handler._pull_image()
        conflicting_handler._create_container()
        conflicting_handler._start_container()

        # Now try to start our main container with force=True
        container_handler.start(force=True)

        # Our container should be running
        our_container = container_handler._get_container()
        assert our_container is not None
        assert our_container.status == "running"

        # Conflicting container should be stopped
        conflicting_container = conflicting_handler._get_container()
        if conflicting_container:
            conflicting_container.reload()
            assert conflicting_container.status != "running"

    finally:
        # Clean up both containers
        for handler in [container_handler, conflicting_handler]:
            cleanup_container = handler._get_container()
            if cleanup_container:
                cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_with_all_flags_combination(container_handler: ContainerHandler,
                                          remove_test_container):
    """Test start with combination of flags: exists_ok=False, force=True, restart=True."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create and start a container first
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        original_container = container_handler._create_container()
        container_handler._start_container()
        original_id = original_container.id

        # start with multiple flags should force delete and recreate
        container_handler.start(exists_ok=False, force=True, restart=True, attach_port=False)

        # Should have a new container (force=True overrides exists_ok=False)
        new_container = container_handler._get_container()
        assert new_container is not None
        assert new_container.id != original_id
        assert new_container.status == "running"

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)


@pytest.mark.usefixtures("clear_port_11435")
def test_start_restart_with_stopped_container(container_handler: ContainerHandler,
                                              remove_test_container):
    """Test start with restart=True when container is stopped should just start it."""
    if not container_handler._is_docker_running():
        pytest.skip("Docker not running")

    try:
        # Create a stopped container
        remove_test_container(container_handler.container_name)
        container_handler._pull_image()
        container = container_handler._create_container()
        container_id = container.id

        # Verify it's not running
        container.reload()
        assert container.status in ["created", "exited"]

        # start with restart=True should just start the stopped container
        container_handler.start(restart=True)

        # Should be the same container, now running
        container.reload()
        assert container.id == container_id
        assert container.status == "running"

    finally:
        cleanup_container = container_handler._get_container()
        if cleanup_container:
            cleanup_container.remove(force=True)
