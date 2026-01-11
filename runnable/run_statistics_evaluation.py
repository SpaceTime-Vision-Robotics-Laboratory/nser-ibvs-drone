import argparse

from auto_follow.evaluation.statistics_evaluation import run_stats
from auto_follow.utils.path_manager import Paths


def main():
    """
    Example of usage:
    python runnable/run_statistics_evaluation.py \
        --results_dir=path/to/real-student-results \
        --scene_name=real-student-results \
        --is_real_world \
        --is_student \

    Scene names:
        - real-ibvs-results
        - real-student-results
        - sim-ibvs-results
        - sim-student-results
    """
    default_save_dir = Paths.BASE_DIR / "results" / "statistics"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="Path to results directory that have testing scenarios")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Directory to save results")
    parser.add_argument("--scene_name", type=str, help="Directory to save results")
    parser.add_argument("--is_real_world", action="store_true", help="If provided will use Real-World configuration and Simulator otherwise.")
    parser.add_argument("--is_student", action="store_true", help="If provided will use Student configuration and NSER-IBVS Teacher otherwise.")
    args = parser.parse_args()

    run_stats(
        base_path=args.results_dir,
        save_path_dir=args.save_dir / args.scene_name,
        is_student=args.is_student,
        is_real=args.is_real_world,
    )

if __name__ == '__main__':
    main()
