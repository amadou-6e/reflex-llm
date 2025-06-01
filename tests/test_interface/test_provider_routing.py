import reflex_llms
from tests.test_interface import *


# Module State Tests
def test_initial_module_state() -> None:
    """Test initial module state is clean."""
    status = reflex_llms.get_module_status()

    assert status["selected_provider"] is None
    assert status["has_cached_config"] is False
    assert status["reflex_server_running"] is False
    assert status["reflex_server_url"] is None


def test_clear_cache_functionality() -> None:
    """Test cache clearing functionality."""
    # Manually set some state to test clearing
    reflex_llms._cached_provider_config = {"test": "config"}
    reflex_llms._selected_provider = "test"

    # Clear cache
    reflex_llms.clear_cache()

    status = reflex_llms.get_module_status()
    assert status["selected_provider"] is None
    assert status["has_cached_config"] is False


def test_get_selected_provider_none() -> None:
    """Test get_selected_provider when no provider selected."""
    assert reflex_llms.get_selected_provider() is None


def test_is_using_reflex_false() -> None:
    """Test is_using_reflex when not using RefLex."""
    assert reflex_llms.is_using_reflex() is False


def test_get_reflex_server_none() -> None:
    """Test get_reflex_server when no RefLex server."""
    assert reflex_llms.get_reflex_server() is None
