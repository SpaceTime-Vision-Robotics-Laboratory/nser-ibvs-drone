import shlex
import time
from contextlib import contextmanager
from pathlib import Path

from auto_follow.simulator.process_management import ProcessManager, SphinxCommandManager
from auto_follow.simulator.sim_config import SimulationConfig, ScriptConfig
from auto_follow.simulator.simulation_process import SimulationProcess, ProcessType
from auto_follow.utils.load_env_file import get_auth_from_vault
from auto_follow.utils.path_manager import Paths
from drone_base.config.logger import LoggerSetup


class SimulationRunner:
    """Controls the Sphinx simulation environment with its firmware and the scripts to execute"""

    def __init__(self, simulator_configuration: SimulationConfig):
        self.config = simulator_configuration
        self._setup_commands()
        self.processes: dict[ProcessType, SimulationProcess | None] = {
            ProcessType.SPHINX: None,
            ProcessType.FIRMWARE: None,
            ProcessType.SCRIPT: None
        }

        self.logger = LoggerSetup.setup_logger(logger_name=self.__class__.__name__)

    def _setup_commands(self):
        self.process_configs = {
            ProcessType.SPHINX: {
                "cmd": shlex.split(self.config.sphinx_cmd),
                "name": ProcessType.SPHINX,
                "init_indicator": "LogApp: Display: Loading level: ",
                "wait_time": self.config.sphinx_init_wait
            },
            ProcessType.FIRMWARE: {
                "cmd": shlex.split(self.config.firmware_command),
                "name": ProcessType.FIRMWARE,
                "init_indicator": "[Msg] All drones instantiated",
                "wait_time": self.config.firmware_init_wait
            }
        }

    def start_process(self, process_type: ProcessType, **kwargs) -> None:
        """Starts a specific process"""
        config = self.process_configs.get(process_type, kwargs)
        process = SimulationProcess(
            cmd=config["cmd"],
            name=config["name"],
            init_indicator=config.get("init_indicator", ""),
            wait_time=config.get("wait_time", 0),
            timeout=self.config.timeout_seconds
        )
        process.start()
        self.processes[process_type] = process

    @contextmanager
    def simulation_session(self, script_config: ScriptConfig | None = None) -> None:
        """Context manager for a simulation session."""
        try:
            self.cleanup()
            self.start_process(ProcessType.SPHINX)
            self.start_process(ProcessType.FIRMWARE)

            if not self.verify_simulation_running():
                raise RuntimeError("Unable to run the simulation session...")

            if script_config is not None:
                self.start_script(script_config)

            yield

        finally:
            self.cleanup(is_verbose=True)

    def cleanup(self, is_verbose: bool = False) -> None:
        """Clean up all processes (in reverse order of their creation) and reset the simulation"""
        if is_verbose:
            self.logger.info("Cleaning up all processes...")

        SphinxCommandManager.drop_all_instances()
        SphinxCommandManager.restart_firmwared(sudo_password=get_auth_from_vault())
        for process in reversed(list(self.processes.values())):
            if process is not None:
                process.stop()

        ProcessManager.clean_remaining_processes(ProcessType.SPHINX)
        time.sleep(5)
        if is_verbose:
            self.logger.info("Cleanup completed.")

    def verify_simulation_running(self) -> bool:
        """Verify if all required processes are  still running."""
        return all(
            ProcessManager.verify_process_running(p.process, p.name) for p in self.processes.values()
            if p is not None and p.name != ProcessType.SCRIPT
        )

    def start_script(self, script_config: ScriptConfig) -> None:
        """Starts a Python script in the simulation environment. Typically, to control a drone"""
        cmd = ["python3", str(script_config.script_path)]
        if script_config.script_args is not None:
            cmd.extend(script_config.script_args)

        self.start_process(ProcessType.SCRIPT, cmd=cmd, name=ProcessType.SCRIPT, wait_time=0)

    def run_simulation(self, script_config: ScriptConfig | None = None) -> bool:
        """Run a complete simulation cycle."""
        try:
            with self.simulation_session(script_config=script_config):
                return self._run_simulation_loop()
        except Exception as e:
            self.logger.error("Simulation failed...", exc_info=e)
            return False

    def _run_simulation_loop(self) -> bool:
        """
        Handles simulation when running with a script (no timeout)
        Runs until the script completes or process terminates.
        """
        self.logger.info("Running simulation with script... Will run until script completes...")
        is_script_over = False
        start_script_time = time.time()
        while not is_script_over:
            if not self.verify_simulation_running():
                self.logger.error("Simulation processes terminated unexpectedly.")
                is_script_over = True
                continue

            script_process = self.processes[ProcessType.SCRIPT]
            if script_process is not None and script_process.process.poll() is not None:
                self.logger.info("Simulation processes terminated successfully.")
                return script_process.process.returncode == 0

            if time.time() - start_script_time > self.config.script_timeout_seconds:
                self.logger.info("Timeout for script execution... Will skip this run...")
                return False

            time.sleep(1)
        return False


def main_simulation_runner(config_path: str | Path = Paths.BUNKER_ANAFI_4K_CONFIG_PATH) -> bool:
    """Runs a simulation cycle and returns if it was executed successfully."""
    from auto_follow.simulator.sim_config import PathConfig

    paths = PathConfig.from_yaml(config_path)
    sim_config = SimulationConfig(
        sphinx_command=paths.sphinx_command,
        firmware_command=paths.firmware_command
    )
    script_config = ScriptConfig(script_path=paths.script_path, script_args=paths.script_args)

    simulation_controller = SimulationRunner(simulator_configuration=sim_config)
    return simulation_controller.run_simulation(script_config=script_config)


def run_simulation_loop(target_runs: int = 2, config_path: str | Path = Paths.BUNKER_ANAFI_4K_CONFIG_PATH):
    total_failures = []
    actual_completed_runs = 0
    run_index = 0
    try:
        while actual_completed_runs != target_runs:
            run_index += 1
            print(f"Starting simulation {run_index}")
            print(f"Configuration file at: {config_path}")
            success = main_simulation_runner(config_path=config_path)
            if success:
                actual_completed_runs += 1
                print(f"Simulation {run_index} completed successfully. "
                      f"(Actual successfully completed {actual_completed_runs}) Waiting 10 seconds...")
            else:
                print(f"Simulation {run_index} failed.")
                total_failures.append(run_index)

            time.sleep(10)
    except KeyboardInterrupt:
        print("\nInterrupted by the user.")
    finally:
        time.sleep(1)
        print(f"Failures indices: {total_failures}")
        print(f"\nFinished simulation runs. Expected {target_runs} successfully ran simulations.\n"
              f"Total failures: {len(total_failures)} out of total of {run_index} runs.")


if __name__ == '__main__':
    target_r = 5
    run_simulation_loop(
        target_runs=target_r, config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-down-left.yaml"
    )
    run_simulation_loop(
        target_runs=target_r, config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-down-right.yaml"
    )
    run_simulation_loop(
        target_runs=target_r, config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-up-left.yaml"
    )
    run_simulation_loop(
        target_runs=target_r, config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-up-right.yaml"
    )
    run_simulation_loop(
        target_runs=target_r, config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-left.yaml"
    )
    run_simulation_loop(
        target_runs=target_r, config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-right.yaml"
    )
    run_simulation_loop(
        target_runs=target_r,
        config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-front-small-offset-left.yaml"
    )
    run_simulation_loop(
        target_runs=target_r,
        config_path=Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-front-small-offset-right.yaml"
    )
