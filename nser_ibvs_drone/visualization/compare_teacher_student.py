from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from nser_ibvs_drone.evaluation.statistics_evaluation import compute_error_statistics, \
    compute_error_statistics_for_time_criteria
from nser_ibvs_drone.visualization.load_data import load_json_flight_data, load_parquet_data, load_json_metadata, \
    ConfigsDirName


class ExpColor:
    PALETTE = {"Teacher": "#2E86AB", "Student": "#A23B72"}


def plot_flight_durations_comparison(teacher_durations_df, student_durations_df,
                                     save_path: str | Path = Path("./plot-output")):
    save_path = Path(save_path)

    teacher_durations_df = teacher_durations_df.copy()
    student_durations_df = student_durations_df.copy()
    teacher_durations_df['method'] = 'Teacher'
    student_durations_df['method'] = 'Student'

    combined_df = pd.concat([teacher_durations_df, student_durations_df], ignore_index=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=combined_df, x="direction", y="flight_duration", hue="method", palette=ExpColor.PALETTE,
                     alpha=0.8)

    for i, patch in enumerate(ax.patches):
        if i >= len(ax.patches) // 2 - 1:
            patch.set_hatch('///')

    plt.title("Flight Duration Comparison: Teacher vs Student")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Flight Duration [seconds]")
    plt.xlabel("Scene Configuration")
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles[:-1]:
        handle.set_hatch(None)
    plt.legend(handles, labels, title="Method")
    plt.tight_layout()
    plt.savefig(save_path / "flight_durations_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=teacher_durations_df, x="flight_duration", label="Teacher", color=ExpColor.PALETTE["Teacher"],
        linewidth=2.5
    )
    sns.kdeplot(
        data=student_durations_df, x="flight_duration", label="Student", color=ExpColor.PALETTE["Student"],
        linewidth=2.5,
        linestyle='--'
    )
    plt.title("Flight Duration Distribution: Teacher vs Student")
    plt.xlabel("Flight Duration [seconds]")
    plt.ylabel("Density")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "flight_durations_histogram_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_distances_comparison(teacher_distances_df, student_distances_df,
                              save_path: str | Path = Path("./plot-output")):
    save_path = Path(save_path)

    teacher_distances_df = teacher_distances_df.copy()
    student_distances_df = student_distances_df.copy()
    teacher_distances_df['method'] = 'Teacher'
    student_distances_df['method'] = 'Student'

    combined_df = pd.concat([teacher_distances_df, student_distances_df], ignore_index=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=combined_df, x="direction", y="distance", hue="method", palette=ExpColor.PALETTE, alpha=0.8)

    for i, patch in enumerate(ax.patches):
        if i >= len(ax.patches) // 2 - 1:
            patch.set_hatch('///')

    plt.title("Distance Comparison: Teacher vs Student")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Scene Configuration")
    plt.ylabel("Distance [meters]")
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles[:-1]:
        handle.set_hatch(None)
    plt.legend(handles, labels, title="Method")
    plt.tight_layout()
    plt.savefig(save_path / "distances_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=teacher_distances_df, x="distance", label="Teacher", color=ExpColor.PALETTE["Teacher"], linewidth=2.5
    )
    sns.kdeplot(
        data=student_distances_df, x="distance", label="Student", color=ExpColor.PALETTE["Student"], linewidth=2.5, linestyle='--'
    )
    plt.title("Distance Distribution: Teacher vs Student")
    plt.xlabel("Distance [meters]")
    plt.ylabel("Density")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "distances_histogram_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_norms_comparison(
        teacher_error_stats_df, student_error_stats_df, save_path: str | Path = Path("./plot-output")
):
    """
    Plot comparison of error norms between teacher and student methods
    """
    save_path = Path(save_path)

    teacher_error_stats_df = teacher_error_stats_df.copy()
    student_error_stats_df = student_error_stats_df.copy()
    teacher_error_stats_df['method'] = 'Teacher'
    student_error_stats_df['method'] = 'Student'

    combined_df = pd.concat([teacher_error_stats_df, student_error_stats_df], ignore_index=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=combined_df, x="direction", y="err_norm_mean", hue="method", palette=ExpColor.PALETTE, alpha=0.8)

    for i, patch in enumerate(ax.patches):
        if i >= len(ax.patches) // 2 - 1:
            patch.set_hatch('///')

    plt.title("Error Norm Comparison: Teacher vs Student (Last 3 Seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Error Norm [pixels]")
    plt.xlabel("Scene Configuration")
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles[:-1]:
        handle.set_hatch(None)
    plt.legend(handles, labels, title="Method")
    plt.tight_layout()
    plt.savefig(save_path / "error_norms_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=teacher_error_stats_df, x="err_norm_mean", label="Teacher", color=ExpColor.PALETTE["Teacher"], linewidth=2.5
    )
    sns.kdeplot(
        data=student_error_stats_df, x="err_norm_mean", label="Student", color=ExpColor.PALETTE["Student"], linewidth=2.5, linestyle='--'
    )
    plt.title("Error Norm Distribution: Teacher vs Student (Last 3 Seconds)")
    plt.xlabel("Mean Error Norm [pixels]")
    plt.ylabel("Density")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "error_norms_histogram_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nError Norm Summary Statistics:")
    print("Teacher Error Norms:")
    print(teacher_error_stats_df[["err_norm_mean", "err_norm_median", "err_norm_std"]].describe())
    print("\nStudent Error Norms:")
    print(student_error_stats_df[["err_norm_mean", "err_norm_median", "err_norm_std"]].describe())


def plot_iou_comparison(teacher_iou_stats_df, student_iou_stats_df,
                        save_path: str | Path = Path("./plot-output")):
    """
    Plot comparison of IoU values between teacher and student methods
    """
    save_path = Path(save_path)

    teacher_iou_stats_df = teacher_iou_stats_df.copy()
    student_iou_stats_df = student_iou_stats_df.copy()
    teacher_iou_stats_df['method'] = 'Teacher'
    student_iou_stats_df['method'] = 'Student'

    combined_df = pd.concat([teacher_iou_stats_df, student_iou_stats_df], ignore_index=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=combined_df, x="direction", y="iou_mean", hue="method", palette=ExpColor.PALETTE, alpha=0.8)

    for i, patch in enumerate(ax.patches):
        if i >= len(ax.patches) // 2 - 1:
            patch.set_hatch('///')

    plt.title("IoU Comparison: Teacher vs Student (Last 3 Seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean IoU")
    plt.xlabel("Scene Configuration")
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles[:-1]:
        handle.set_hatch(None)
    plt.legend(handles, labels, title="Method")
    plt.tight_layout()
    plt.savefig(save_path / "iou_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=teacher_iou_stats_df, x="iou_mean", label="Teacher", color=ExpColor.PALETTE["Teacher"], linewidth=2.5
    )
    sns.kdeplot(
        data=student_iou_stats_df, x="iou_mean", label="Student", color=ExpColor.PALETTE["Student"], linewidth=2.5,
        linestyle='--'
    )
    plt.title("IoU Distribution: Teacher vs Student (Last 3 Seconds)")
    plt.xlabel("Mean IoU")
    plt.ylabel("Density")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "iou_histogram_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nIoU Summary Statistics:")
    print("Teacher IoU:")
    print(teacher_iou_stats_df[["iou_mean", "iou_median", "iou_std"]].describe())
    print("\nStudent IoU:")
    print(student_iou_stats_df[["iou_mean", "iou_median", "iou_std"]].describe())


def plot_command_distribution_magnitude_comparison(
        teacher_parquet_df: pd.DataFrame, student_parquet_df: pd.DataFrame,
        save_path: str | Path = Path("./plot-output")
):
    save_path = Path(save_path)

    all_data = pd.concat([teacher_parquet_df, student_parquet_df])
    x_min, x_max = all_data["x_cmd"].min(), all_data["x_cmd"].max()
    y_min, y_max = all_data["y_cmd"].min(), all_data["y_cmd"].max()
    rot_min, rot_max = all_data["rot_cmd"].min(), all_data["rot_cmd"].max()

    num_bins = 50
    x_bins = np.linspace(x_min, x_max, num_bins + 1)
    y_bins = np.linspace(y_min, y_max, num_bins + 1)
    rot_bins = np.linspace(rot_min, rot_max, num_bins + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(data=teacher_parquet_df, x="x_cmd", kde=True, bins=x_bins, alpha=0.7, label="Teacher", color=ExpColor.PALETTE["Teacher"])
    sns.histplot(data=student_parquet_df, x="x_cmd", kde=True, bins=x_bins, alpha=0.5, label="Student", color=ExpColor.PALETTE["Student"])
    plt.title("Distribution of x_cmd: Teacher vs Student")
    plt.xlabel("x_cmd")
    plt.ylabel("Density")
    plt.legend()

    plt.subplot(1, 3, 2)
    sns.histplot(data=teacher_parquet_df, x="y_cmd", kde=True, bins=y_bins, alpha=0.7, label="Teacher", color=ExpColor.PALETTE["Teacher"])
    sns.histplot(data=student_parquet_df, x="y_cmd", kde=True, bins=y_bins, alpha=0.5, label="Student", color=ExpColor.PALETTE["Student"])
    plt.title("Distribution of y_cmd: Teacher vs Student")
    plt.xlabel("y_cmd")
    plt.ylabel("Density")
    plt.legend()

    plt.subplot(1, 3, 3)
    sns.histplot(data=teacher_parquet_df, x="rot_cmd", kde=True, bins=rot_bins, alpha=0.7, label="Teacher", color=ExpColor.PALETTE["Teacher"])
    sns.histplot(data=student_parquet_df, x="rot_cmd", kde=True, bins=rot_bins, alpha=0.5, label="Student", color=ExpColor.PALETTE["Student"])
    plt.title("Distribution of rot_cmd: Teacher vs Student")
    plt.xlabel("rot_cmd")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path / "command_distributions_magnitude_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nTeacher Summary Statistics:")
    print(teacher_parquet_df[["x_cmd", "y_cmd", "rot_cmd"]].describe())
    print("\nStudent Summary Statistics:")
    print(student_parquet_df[["x_cmd", "y_cmd", "rot_cmd"]].describe())


def plot_command_distribution_comparison(teacher_parquet_df: pd.DataFrame, student_parquet_df: pd.DataFrame,
                                         save_path: str | Path = Path("./plot-output")):
    save_path = Path(save_path)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.kdeplot(data=teacher_parquet_df, x="x_cmd", label="Teacher", color=ExpColor.PALETTE["Teacher"], linewidth=2.5)
    sns.kdeplot(data=student_parquet_df, x="x_cmd", label="Student", color=ExpColor.PALETTE["Student"], linewidth=2.5, linestyle='--')
    plt.title("Distribution of x_cmd: Teacher vs Student")
    plt.xlabel("x_cmd")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    sns.kdeplot(data=teacher_parquet_df, x="y_cmd", label="Teacher", color=ExpColor.PALETTE["Teacher"], linewidth=2.5)
    sns.kdeplot(data=student_parquet_df, x="y_cmd", label="Student", color=ExpColor.PALETTE["Student"], linewidth=2.5, linestyle='--')
    plt.title("Distribution of y_cmd: Teacher vs Student")
    plt.xlabel("y_cmd")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    sns.kdeplot(data=teacher_parquet_df, x="rot_cmd", label="Teacher", color=ExpColor.PALETTE["Teacher"], linewidth=2.5)
    sns.kdeplot(data=student_parquet_df, x="rot_cmd", label="Student", color=ExpColor.PALETTE["Student"], linewidth=2.5, linestyle='--')
    plt.title("Distribution of rot_cmd: Teacher vs Student")
    plt.xlabel("rot_cmd")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "command_distributions_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nTeacher Summary Statistics:")
    print(teacher_parquet_df[["x_cmd", "y_cmd", "rot_cmd"]].describe())
    print("\nStudent Summary Statistics:")
    print(student_parquet_df[["x_cmd", "y_cmd", "rot_cmd"]].describe())


def run_comparison_analysis(
        teacher_base_path: str | Path,
        student_base_path: str | Path,
        save_path_name: str | Path,
        is_real: bool = False
):
    """
    Run comparison analysis between teacher and student methods
    """
    reset_matplotlib_state()

    teacher_base_path = Path(teacher_base_path)
    student_base_path = Path(student_base_path)
    save_path_dir = Path(save_path_name)
    save_path_dir.mkdir(parents=True, exist_ok=True)

    scenes_names = ConfigsDirName.REAL if is_real else ConfigsDirName.SIM

    student_scenes_names = scenes_names.copy()
    for index, scene_name in enumerate(student_scenes_names):
        if is_real:
            student_scenes_names[index] = scene_name.replace("ibvs", "student")
        else:
            student_scenes_names[index] = f"{scene_name}-student"

    print("Loading teacher data...")
    teacher_json_flight_data = load_json_flight_data(teacher_base_path, scenes_names)
    teacher_metadata_df = load_json_metadata(teacher_base_path, scenes_names)
    teacher_parquet_data = load_parquet_data(teacher_base_path, scenes_names, is_student=False)

    print("Loading student data...")
    student_json_flight_data = load_json_flight_data(student_base_path, student_scenes_names)
    student_metadata_df = load_json_metadata(student_base_path, student_scenes_names)
    student_parquet_data = load_parquet_data(student_base_path, student_scenes_names, is_student=True)

    print("Computing error statistics...")
    teacher_error_stats, teacher_iou_stats = compute_error_statistics(teacher_parquet_data, teacher_json_flight_data)
    if is_real:
        student_error_stats, student_iou_stats = compute_error_statistics(student_parquet_data,
                                                                          student_json_flight_data)
        plot_flight_durations_comparison(teacher_json_flight_data, student_json_flight_data, save_path=save_path_dir)
    else:
        student_error_stats, student_iou_stats, updated_student_time_data = compute_error_statistics_for_time_criteria(
            student_parquet_data, student_json_flight_data, threshold=1.0, duration=3.0
        )
        plot_flight_durations_comparison(teacher_json_flight_data, updated_student_time_data, save_path=save_path_dir)

    print(
        f"Teacher - Total scenes: {teacher_parquet_data['scene'].nunique()}, Total runs: {teacher_parquet_data['run'].nunique()}")
    print(
        f"Student - Total scenes: {student_parquet_data['scene'].nunique()}, Total runs: {student_parquet_data['run'].nunique()}")
    print(f"Teacher - Error stats: {len(teacher_error_stats)} runs, IoU stats: {len(teacher_iou_stats)} runs")
    print(f"Student - Error stats: {len(student_error_stats)} runs, IoU stats: {len(student_iou_stats)} runs")

    print("Creating comparison plots...")
    plot_distances_comparison(teacher_metadata_df, student_metadata_df, save_path=save_path_dir)
    plot_error_norms_comparison(teacher_error_stats, student_error_stats, save_path=save_path_dir)
    plot_iou_comparison(teacher_iou_stats, student_iou_stats, save_path=save_path_dir)

    plot_command_distribution_magnitude_comparison(teacher_parquet_data, student_parquet_data, save_path=save_path_dir)
    plot_command_distribution_comparison(teacher_parquet_data, student_parquet_data, save_path=save_path_dir)
    print(f"Comparison plots saved to: {save_path_dir}")


def reset_matplotlib_state():
    """Reset matplotlib to clean state between plotting sessions"""
    plt.clf()
    plt.cla()
    plt.close('all')
    plt.rcdefaults()
    sns.reset_defaults()


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    teacher_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/ibvs/car-ibvs-sim-results-good"
    student_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student-with-teacher-output/sim-student-with-teacher-output-pc-sebi"

    teacher_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/ibvs/sim-ibvs-results-merged"
    student_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student-with-teacher-output/sim-student-with-teacher-output-merged"

    try:
        run_comparison_analysis(
            teacher_base_path=Path(teacher_path),
            student_base_path=Path(student_path),
            save_path_name="teacher_student_comparison-sim_v2",
            is_real=False,
        )
        print("Comparison analysis completed successfully!")
    except Exception as e:
        print(f"Error in comparison analysis: {e}")

    teacher_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results"
    teacher_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged"
    student_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/student/real-student-with-teacher-output-new/results-real-student-21-06"

    try:
        run_comparison_analysis(
            teacher_base_path=Path(teacher_path),
            student_base_path=Path(student_path),
            save_path_name="teacher_student_comparison-real",
            is_real=True,
        )
        print("Comparison analysis completed successfully!")
    except Exception as e:
        print(f"Error in comparison analysis: {e}")
