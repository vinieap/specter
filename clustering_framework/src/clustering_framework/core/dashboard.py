"""Dashboard process management."""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

import psutil


class ProcessState(Enum):
    """Process states."""

    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    FAILED = auto()


@dataclass
class ProcessMetrics:
    """Process performance metrics."""

    cpu_percent: float
    memory_usage: int  # In bytes
    thread_count: int
    open_files: int


class ProcessManager:
    """Process manager for dashboard server.

    This class handles starting, stopping, monitoring, and restarting
    the dashboard process.
    """

    def __init__(
        self,
        check_interval: float = 1.0,
        restart_delay: float = 5.0,
        max_restarts: int = 3,
    ):
        """Initialize process manager.

        Args:
            check_interval: Interval between health checks in seconds
            restart_delay: Delay before restart attempts in seconds
            max_restarts: Maximum number of restart attempts
        """
        self._state = ProcessState.STOPPED
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._start_time: Optional[float] = None
        self._metrics: Optional[ProcessMetrics] = None
        self._restart_count = 0
        self._temp_files: Set[str] = set()
        self._cleanup_hooks: List[Callable[[], None]] = []

        self._check_interval = check_interval
        self._restart_delay = restart_delay
        self._max_restarts = max_restarts

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()

    def start_dashboard(self, port: int = 8080) -> None:
        """Start the dashboard process.

        Args:
            port: Port number for the dashboard server

        Raises:
            RuntimeError: If process is already running or start fails
        """
        with self._lock:
            if self._process is not None or self._state != ProcessState.STOPPED:
                raise RuntimeError("Dashboard process is already running")

            self._state = ProcessState.STARTING
            self._port = port
            self._start_time = time.time()

            try:
                # Start dashboard process
                self._process = subprocess.Popen(
                    [
                        "optuna-dashboard",
                        "sqlite:///optuna_clustering_studies.db",
                        f"--port={port}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

                # Start monitoring
                self._stop_monitoring.clear()
                self._monitor_thread = threading.Thread(
                    target=self._monitor_process,
                    daemon=True,
                )
                self._monitor_thread.start()

                self._state = ProcessState.RUNNING

            except Exception as e:
                self._state = ProcessState.FAILED
                self._restart_count += 1
                if self._restart_count >= self._max_restarts:
                    raise RuntimeError("Maximum restart attempts reached")
                raise RuntimeError(f"Failed to start dashboard: {e}")

    def stop_dashboard(self, timeout: float = 5.0) -> None:
        """Stop the dashboard process.

        Args:
            timeout: Timeout for graceful shutdown in seconds
        """
        with self._lock:
            if self._state == ProcessState.STOPPED:
                return

            # Stop monitoring
            if self._monitor_thread is not None:
                self._stop_monitoring.set()
                try:
                    self._monitor_thread.join(timeout=1.0)
                except RuntimeError:
                    pass  # Ignore join errors during cleanup

            # Stop process
            if self._process is not None:
                try:
                    # Send SIGTERM to process group
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill if timeout
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already dead

            self._cleanup()
            self._state = ProcessState.STOPPED
            self._process = None
            self._port = None
            self._start_time = None
            self._metrics = None

    def restart_dashboard(self) -> None:
        """Restart the dashboard process.

        Raises:
            RuntimeError: If maximum restart attempts reached
        """
        if self._restart_count > self._max_restarts:
            raise RuntimeError("Maximum restart attempts reached")

        self.stop_dashboard()
        time.sleep(self._restart_delay)
        self.start_dashboard(port=self._port or 8080)

    def get_status(self) -> Dict[str, Any]:
        """Get current process status.

        Returns:
            Dictionary containing process status and metrics
        """
        status = {
            "state": self._state.name.lower(),
            "port": self._port,
            "uptime": (
                time.time() - self._start_time
                if self._start_time is not None
                else 0
            ),
            "restart_count": self._restart_count,
            "metrics": None,
        }

        if self._metrics is not None:
            status["metrics"] = {
                "cpu_percent": self._metrics.cpu_percent,
                "memory_usage": self._metrics.memory_usage,
                "thread_count": self._metrics.thread_count,
                "open_files": self._metrics.open_files,
            }

        return status

    def register_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """Register a cleanup hook.

        Args:
            hook: Function to call during cleanup
        """
        self._cleanup_hooks.append(hook)

    def add_temp_file(self, file_path: str) -> None:
        """Register a temporary file for cleanup.

        Args:
            file_path: Path to temporary file
        """
        self._temp_files.add(file_path)

    def cleanup(self) -> None:
        """Run cleanup operations."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Internal cleanup helper."""
        # Run cleanup hooks
        for hook in self._cleanup_hooks:
            try:
                hook()
            except Exception:
                pass  # Ignore cleanup errors

        # Remove temp files
        for file_path in self._temp_files:
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore file removal errors

        self._temp_files.clear()

    def _monitor_process(self) -> None:
        """Monitor process health."""
        while not self._stop_monitoring.is_set():
            try:
                # Check if process is alive
                if self._process is None or self._process.poll() is not None:
                    with self._lock:
                        self._state = ProcessState.FAILED
                        self._restart_count += 1
                    break

                # Collect metrics
                process = psutil.Process(self._process.pid)
                self._metrics = ProcessMetrics(
                    cpu_percent=process.cpu_percent(),
                    memory_usage=process.memory_info().rss,
                    thread_count=process.num_threads(),
                    open_files=len(process.open_files()),
                )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                with self._lock:
                    self._state = ProcessState.FAILED
                    self._restart_count += 1
                break

            time.sleep(self._check_interval)

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_dashboard()