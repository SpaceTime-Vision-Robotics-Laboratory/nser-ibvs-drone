import argparse

from auto_follow.utils.path_manager import Paths
from auto_follow.visualization.visualize_distributions import run_plot_analysis_on_scenes


def main():
    """
    Generates plots about flight for a given scene experiments.

    If the scene experiments come from the Student method it needs Teacher logs generation
    use auto_follow/evaluation/ibvs_splitter_run_for_logs.py to generate them if needed.

    Example of usage:
    python runnable/visualization/run_method_distributions_analysis.py \
        --base_experiments_dir=/path/to/experiments/ \
        --save_dir=/path/to/save-directory/ \
        --is_real_world \
        --is_student \
        --random_runs=1
    """
    save_dir = Paths.BASE_DIR / "results" / "plots"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_experiments_dir", type=str, help="Path to experiments directory")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Path to save the results")
    parser.add_argument("--is_real_world", action="store_true", help="If provided will use Real-World configuration and Simulator otherwise.")
    parser.add_argument("--is_student", action="store_true", help="If provided will use Student configuration and NSER-IBVS Teacher otherwise.")
    parser.add_argument("--random_runs", type=int, default=5, help="Number of random runs to generate additional plots.")
    args = parser.parse_args()

    try:
        run_plot_analysis_on_scenes(
            base_path=args.base_experiments_dir,
            save_path_name=args.save_dir,
            is_student=args.is_student,
            is_real=args.is_real_world,
            random_runs=args.random_runs
        )
    except Exception as e:
        print(f"SIM prob: {e}")

if __name__ == '__main__':
    main()