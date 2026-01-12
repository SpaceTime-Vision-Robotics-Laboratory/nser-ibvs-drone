import argparse
from pathlib import Path

from nser_ibvs_drone.utils.path_manager import Paths
from nser_ibvs_drone.visualization.compare_teacher_student import run_comparison_analysis


def main():
    """
    Runs comparison analysis using teacher and student experiments to generate plots.

    Example of usage:
    python runnable/visualization/run_teacher_student_comparison.py \
        --teacher_exp_path=/path/to/teacher/experiments \
        --student_exp_path=/path/to/student/experiments \
        --save_dir=/path/to/save/directory
        --is_real_world
    """
    save_dir = Paths.BASE_DIR / "results" / "plots"

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_exp_path", type=str, help="Path to teacher NSER-IBVS experiments base directory")
    parser.add_argument("--student_exp_path", type=str, help="Path to student experiments base directory")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Path to save the results")
    parser.add_argument("--is_real_world", action="store_true", help="If provided will use Real-World configuration and Simulator otherwise.")

    args = parser.parse_args()

    save_path_name = "teacher-student-comparison-"
    if args.is_real_world:
        save_path_name = save_path_name + "real-world"
    else:
        save_path_name = save_path_name + "simulator"

    try:
        run_comparison_analysis(
            teacher_base_path=Path(args.teacher_exp_path),
            student_base_path=Path(args.student_exp_path),
            save_path_name=save_dir / save_path_name,
            is_real=args.is_real_world,
        )
        print("Comparison analysis completed successfully!")
    except Exception as e:
        print(f"Error in comparison analysis: {e}")


if __name__ == '__main__':
    main()
