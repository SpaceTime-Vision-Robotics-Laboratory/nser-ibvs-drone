import argparse

from auto_follow.simulator.simulation_loop_manager import SimulationLoopManager


def main():
    parser = argparse.ArgumentParser(description="Run simulator experiments suite on 8 available scenes.")
    parser.add_argument(
        "--sphinx_bunker_base_dir", type=str,
        help="Path to where the Sphinx Bunker UE4 application exists on your machine"
    )
    parser.add_argument(
        "--target_runs", type=int, default=1,
        help="Number of times to run an experiment on a scene."
    )
    parser.add_argument(
        "--is_student", action="store_true",
        help="If the method to run is the Distilled Student version or IBVS Splitter"
    )
    args = parser.parse_args()

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
        sphinx_base_dir=args.sphinx_bunker_base_dir,
        target_runs=args.target_runs,
        is_student=args.is_student,
    )
    sim_loop_manager.run_suite(scenes_to_run)


if __name__ == '__main__':
    main()
