import time

from auto_follow.utils.path_manager import Paths
from drone_base.config.logger import LoggerSetup
from drone_sim_runner.sphx.simulation_runner import main_simulation_runner


class SimulationLoopManager:

    def __init__(self, sphinx_base_dir: str, target_runs: int, is_student: bool):
        self.sphinx_base_dir = sphinx_base_dir
        self.target_runs = target_runs
        self.is_student = is_student

        self.failures = []
        self.total_attempts = 0
        self.successful_runs = 0
        self.delay_between_runs = 10

        self.logger = LoggerSetup.setup_logger(logger_name=self.__class__.__name__)

    def _prepare_scene_name(self, scene_name: str) -> str:
        """Appends student suffix if mode is enabled"""
        if self.is_student and not scene_name.endswith('-student.yaml'):
            return scene_name.replace(".yaml", "-student.yaml")
        return scene_name

    def run_single_config(self, scene_name: str):
        """Executes the simulation loop for a specific scene configuration"""
        scene_name = self._prepare_scene_name(scene_name)
        config_path = Paths.SIMULATOR_CONFIG_DIR / scene_name
        self.logger.info("Starting Session: %s", scene_name)
        current_scene_successful_runs = 0

        while current_scene_successful_runs < self.target_runs:
            self.total_attempts += 1
            self.logger.info("Attempt %s | Scene: %s", self.total_attempts, scene_name)

            success = main_simulation_runner(
                sphinx_base_dir=self.sphinx_base_dir,
                config_path=config_path,
                projects_base_dir=Paths.BASE_DIR,
                output_dir=Paths.OUTPUT_DIR,
                auth_path=Paths.BASE_DIR / ".env"
            )

            if success:
                current_scene_successful_runs += 1
                self.successful_runs += 1
                self.logger.info("Success! (%s/%s)", current_scene_successful_runs, self.target_runs)
            else:
                self.failures.append(self.total_attempts)
                self.logger.warning("Simulation failed at attempt %s", self.total_attempts)

            if current_scene_successful_runs < self.target_runs:
                time.sleep(self.delay_between_runs)

    def run_suite(self, scenes: list[str]):
        """Runs the experiments on a list of different scenes."""
        try:
            for scene in scenes:
                self.run_single_config(scene)
        except KeyboardInterrupt:
            self.logger.warning("Simulation interrupted by user.")
        finally:
            self.print_summary()

    def print_summary(self):
        print("\n\n" + "=" * 30)
        print("Completed! Simulation Summary:")
        print(f"\t- Total Attempts: {self.total_attempts}")
        print(f"\t- Successful Runs: {self.successful_runs}")
        print(f"\t- Total Failures: {len(self.failures)}")
        print(f"\t- Failure Indices: {self.failures}")


if __name__ == '__main__':
    sphinx_bunker_base_dir = "/home/brittle/Games/MyGames/DroneSimulation"
    scenes_to_run = [
        "bunker-online-4k-config-test-down-left.yaml",
        "bunker-online-4k-config-test-down-right.yaml",
        "bunker-online-4k-config-test-front-small-offset-right.yaml",
        "bunker-online-4k-config-test-front-small-offset-left.yaml",
        "bunker-online-4k-config-test-left.yaml",
        "bunker-online-4k-config-test-right.yaml",
        "bunker-online-4k-config-test-up-left.yaml",
        "bunker-online-4k-config-test-up-right.yaml",
    ]
    sim_loop_manager = SimulationLoopManager(
        sphinx_base_dir=sphinx_bunker_base_dir,
        target_runs=2,
        is_student=False,
    )
    sim_loop_manager.run_suite(scenes_to_run)
