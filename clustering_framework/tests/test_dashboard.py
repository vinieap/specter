"""Tests for dashboard process management."""

import signal
import subprocess
import time
from unittest.mock import MagicMock, patch

import psutil
import pytest

from clustering_framework.core.dashboard import (
    ProcessState,
    ProcessManager,
)


@pytest.fixture
def mock_process():
    """Create a mock subprocess.Popen instance."""
    process = MagicMock(spec=subprocess.Popen)
    process.pid = 12345
    process.poll.return_value = None
    process.wait = MagicMock()  # Mock wait to do nothing
    return process


@pytest.fixture
def mock_psutil_process():
    """Create a mock psutil.Process instance."""
    process = MagicMock(spec=psutil.Process)
    process.cpu_percent.return_value = 1.5
    process.memory_info().rss = 1024 * 1024 * 100  # 100 MB
    process.open_files.return_value = []
    process.num_threads.return_value = 5
    return process


def test_process_lifecycle(mock_process):
    """Test basic process lifecycle management."""
    with patch("subprocess.Popen", return_value=mock_process):
        manager = ProcessManager()

        # Test starting
        manager.start_dashboard(port=8080)
        assert manager._state == ProcessState.RUNNING
        assert manager._port == 8080
        assert manager._process is not None

        # Test stopping
        manager.stop_dashboard()
        assert manager._state == ProcessState.STOPPED
        assert manager._process is None


def test_process_monitoring(mock_process, mock_psutil_process):
    """Test process health monitoring."""
    with patch("subprocess.Popen", return_value=mock_process), patch(
        "psutil.Process", return_value=mock_psutil_process
    ):
        manager = ProcessManager(check_interval=0.1)
        manager.start_dashboard()

        # Wait for metrics collection
        time.sleep(0.2)
        status = manager.get_status()

        assert status["state"] == "running"
        assert status["metrics"] is not None
        assert status["metrics"]["cpu_percent"] == 1.5
        assert status["metrics"]["thread_count"] == 5

        manager.stop_dashboard()


def test_process_failure_handling(mock_process):
    """Test handling of process failures."""
    with patch("subprocess.Popen") as mock_popen, patch(
        "threading.Thread"
    ) as mock_thread, patch("time.sleep") as mock_sleep:
        # Configure mock process for initial start
        mock_popen.return_value = mock_process
        mock_process.poll.return_value = None  # Initially running

        # Configure mock thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        def mock_thread_start():
            if hasattr(mock_thread_instance, "_target"):
                mock_thread_instance._target()

        mock_thread_instance.start = mock_thread_start
        mock_thread_instance.join = MagicMock()  # Mock join to do nothing

        # Configure sleep to do nothing
        mock_sleep.return_value = None

        manager = ProcessManager(
            check_interval=0.1,
            restart_delay=0.1,
            max_restarts=2,
        )

        # Start dashboard - process should be running
        manager.start_dashboard()
        assert manager._state == ProcessState.RUNNING
        assert manager._restart_count == 0

        # Simulate first failure
        mock_process.poll.return_value = 1  # Process dies
        manager._monitor_process()  # Detect failure
        assert manager._restart_count == 1
        assert manager._state == ProcessState.FAILED

        # Mock successful restart
        mock_process.poll.return_value = None  # Process running again
        mock_popen.reset_mock()  # Reset for next Popen call
        mock_popen.return_value = mock_process  # Set up next process

        manager.restart_dashboard()  # Attempt restart
        assert manager._state == ProcessState.RUNNING

        # Simulate second failure
        mock_process.poll.return_value = 1  # Process dies again
        manager._monitor_process()  # Detect failure
        assert manager._restart_count == 2
        assert manager._state == ProcessState.FAILED

        # Mock successful restart
        mock_process.poll.return_value = None  # Process running again
        mock_popen.reset_mock()  # Reset for next Popen call
        mock_popen.return_value = mock_process  # Set up next process

        manager.restart_dashboard()  # Attempt restart
        assert manager._state == ProcessState.RUNNING

        manager.stop_dashboard()


def test_cleanup_hooks():
    """Test cleanup hook execution."""
    cleanup_called = []

    def cleanup_hook():
        cleanup_called.append(True)

    manager = ProcessManager()
    manager.register_cleanup_hook(cleanup_hook)
    manager.cleanup()

    assert len(cleanup_called) == 1


def test_temp_file_cleanup(tmp_path):
    """Test temporary file cleanup."""
    # Create temp file
    temp_file = tmp_path / "test.tmp"
    temp_file.write_text("test")
    assert temp_file.exists()

    # Register for cleanup
    manager = ProcessManager()
    manager.add_temp_file(str(temp_file))
    manager.cleanup()

    assert not temp_file.exists()


def test_graceful_shutdown(mock_process):
    """Test graceful process shutdown."""
    with patch("subprocess.Popen", return_value=mock_process), patch(
        "os.killpg"
    ) as mock_kill, patch("os.getpgid", return_value=12345):
        manager = ProcessManager()
        manager.start_dashboard()

        # Test graceful shutdown
        manager.stop_dashboard(timeout=1.0)
        mock_kill.assert_called_with(12345, signal.SIGTERM)

        # Process should stop within timeout
        mock_process.wait.assert_called_with(timeout=1.0)


def test_restart_limits():
    """Test restart attempt limits."""

    def failing_popen(*args, **kwargs):
        raise RuntimeError("Failed to start")

    with patch("subprocess.Popen", side_effect=failing_popen):
        manager = ProcessManager(max_restarts=1)

        # First attempt
        with pytest.raises(RuntimeError):
            manager.start_dashboard()

        # Second attempt (should hit limit)
        with pytest.raises(RuntimeError, match="Maximum restart attempts reached"):
            manager.restart_dashboard()


def test_concurrent_operations(mock_process):
    """Test concurrent operation handling."""
    with patch("subprocess.Popen", return_value=mock_process):
        manager = ProcessManager()
        manager.start_dashboard()

        # Try starting while running
        with pytest.raises(RuntimeError, match="already running"):
            manager.start_dashboard()

        # Stop should work
        manager.stop_dashboard()
        assert manager._state == ProcessState.STOPPED

        # Multiple stops should be safe
        manager.stop_dashboard()
        manager.stop_dashboard()


def test_status_reporting(mock_process, mock_psutil_process):
    """Test status and metrics reporting."""
    with patch("subprocess.Popen", return_value=mock_process), patch(
        "psutil.Process", return_value=mock_psutil_process
    ):
        manager = ProcessManager(check_interval=0.1)

        # Check initial status
        status = manager.get_status()
        assert status["state"] == "stopped"
        assert status["metrics"] is None
        assert status["restart_count"] == 0

        # Start and check running status
        manager.start_dashboard(port=8080)
        time.sleep(0.2)  # Wait for metrics
        status = manager.get_status()
        assert status["state"] == "running"
        assert status["port"] == 8080
        assert status["metrics"] is not None
        assert status["uptime"] > 0

        manager.stop_dashboard()
