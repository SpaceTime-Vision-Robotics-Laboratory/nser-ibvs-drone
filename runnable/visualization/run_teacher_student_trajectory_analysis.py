import argparse

from nser_ibvs_drone.utils.path_manager import Paths
from nser_ibvs_drone.visualization.metadata_trajectory_plot_compare_teacher_student import generate_experiments_paths, \
    process_comparison_multiple_metadata_files


def main():
    """
    Generates all trajectories comparison between Teacher and Student methods from the given paths.

    Example of usage:
    python runnable/visualization/run_teacher_student_trajectory_analysis.py \
        --teacher_exp_path=/path/to/simulated-teacher-experiments/ \
        --student_exp_path=/path/to/simulated-student-experiments/
    """
    save_dir = Paths.BASE_DIR / "results" / "trajectories"
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_exp_path", type=str, help="Path to teacher NSER-IBVS experiments base directory")
    parser.add_argument("--student_exp_path", type=str, help="Path to student experiments base directory")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Path to save the results")
    args = parser.parse_args()

    goal = (2, 2.5)
    front_paths = generate_experiments_paths(
        base_path_teacher=args.teacher_exp_path,
        base_path_student=args.student_exp_path,
        goal=goal,
        direction_pair="front"
    )
    process_comparison_multiple_metadata_files(front_paths, goal=goal, direction_pair="front", save_dir=args.save_dir)

    up_paths = generate_experiments_paths(
        base_path_teacher=args.teacher_exp_path,
        base_path_student=args.student_exp_path,
        goal=goal,
        direction_pair="up"
    )
    process_comparison_multiple_metadata_files(up_paths, goal=goal, direction_pair="up", save_dir=args.save_dir)

    left_right_paths = generate_experiments_paths(
        base_path_teacher=args.teacher_exp_path,
        base_path_student=args.student_exp_path,
        goal=goal,
        direction_pair="left-right"
    )
    process_comparison_multiple_metadata_files(left_right_paths, goal=goal, direction_pair="left-right",
                                               save_dir=args.save_dir)

    down_paths = generate_experiments_paths(
        base_path_teacher=args.teacher_exp_path,
        base_path_student=args.student_exp_path,
        goal=goal,
        direction_pair="down"
    )
    process_comparison_multiple_metadata_files(down_paths, goal=goal, direction_pair="down", save_dir=args.save_dir)


if __name__ == '__main__':
    main()
