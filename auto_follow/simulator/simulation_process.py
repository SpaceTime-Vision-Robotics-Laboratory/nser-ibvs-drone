import subprocess
import threading
import time

from auto_follow.simulator.process_management import ProcessManager
from auto_follow.simulator.process_type import ProcessType
from drone_base.config.logger import LoggerSetup


class SimulationProcess:
    """Represents a single process in the simulation."""

    def __init__(self, cmd: list[str], name: ProcessType, init_indicator: str, wait_time: int, timeout: int):
        self.cmd = cmd
        self.name = name
        self.init_indicator = init_indicator
        self.wait_time = wait_time
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.output_thread = None
        self.is_running = False

        self.logger = LoggerSetup.setup_logger(logger_name=self.__class__.__name__)

    def _log_output(self):
        while self.is_running and self.process is not None and self.process.stdout:
            try:
                output = self.process.stdout.readline()
                if output is not None:
                    output = output.decode("utf-8").strip()
                    if self.name == ProcessType.SCRIPT and output != "":
                        self.logger.info("Script output: %s", output)

            except Exception as e:
                self.logger.error("Exception while reading '%s' output...", self.name, exc_info=e)

    def start(self, buffer_size: int = -1) -> None:
        """Start the process and wait for initialization to complete."""
        self.is_running = True
        try:
            self.logger.info("Starting '%s' process...", self.name)
            self.process = subprocess.Popen(
                ["stdbuf", "-o0", *self.cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=buffer_size,
            )
            if self.name == ProcessType.SCRIPT:
                self.output_thread = threading.Thread(target=self._log_output, daemon=True)
                self.output_thread.start()
            else:
                self._wait_for_initialization()
            time.sleep(self.wait_time)
            self.logger.info("Process '%s' started successfully.", self.name)
        except Exception as e:
            raise RuntimeError(f"Failed to start '{self.name}'") from e

    def _wait_for_initialization(self) -> None:
        """Wait for process initialization to complete."""
        is_initialized = False
        start_time = time.time()
        while not is_initialized:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timeout while waiting for '{self.name}' process to initialize successfully.")

            if not ProcessManager.verify_process_running(self.process, self.name):
                raise RuntimeError(f"Process '{self.name}' terminated unexpectedly.")

            output = self.process.stdout.readline().decode("utf-8").strip()
            if output:
                self.logger.debug("'%s' output: %s", self.name, output)
                if self.init_indicator in output:
                    self.logger.info("Found '%s' initialization indicator: %s", self.name, self.init_indicator)
                    is_initialized = True

    def stop(self) -> None:
        """Kills the process."""
        self.is_running = False
        if self.process is not None:
            ProcessManager.kill_process_tree(self.process.pid)
            self.process = None

        if self.name == ProcessType.SCRIPT and self.output_thread is not None and self.output_thread.is_alive():
            self.output_thread.join(timeout=1)
