import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from auto_follow.visualization.load_data import load_json_flight_data, load_parquet_data, load_json_metadata, \
    ConfigsDirName


def points_to_bbox(points):
    xs = points[::2]
    ys = points[1::2]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return x_min, y_min, x_max, y_max


def compute_iou(current_box, goal_box):
    x_a = max(current_box[0], goal_box[0])
    y_a = max(current_box[1], goal_box[1])
    x_b = min(current_box[2], goal_box[2])
    y_b = min(current_box[3], goal_box[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    current_box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
    goal_box_area = (goal_box[2] - goal_box[0]) * (goal_box[3] - goal_box[1])
    union_area = current_box_area + goal_box_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_error_statistics(parquet_df: pd.DataFrame, duration_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    err_norm_stats = []
    iou_stats = []

    for run_id, run_group in parquet_df.groupby("run"):
        duration_row = duration_df[duration_df["run"] == run_id]
        if duration_row.empty:
            continue

        flight_end = duration_row.iloc[0]["end_time"]
        direction = duration_row.iloc[0]["direction"]
        time_cutoff = flight_end - 3  # last 3 seconds

        run_last3s = run_group[run_group["timestamp"] >= time_cutoff]

        if run_last3s.empty:
            continue

        err_uv_norms = run_last3s["err_uv"].apply(lambda v: np.linalg.norm(v))
        err_norm_stats.append({
            "run": run_id,
            "direction": direction,
            "err_norm_mean": err_uv_norms.mean(),
            "err_norm_median": err_uv_norms.median(),
            "err_norm_std": err_uv_norms.std(),
            "err_norm_min": err_uv_norms.min(),
            "err_norm_max": err_uv_norms.max(),
        })

        run_iou = []
        for _, row, in run_last3s.iterrows():
            curr_points = row["current_points_flatten"]
            goal_points = row["goal_points_flatten"]
            if curr_points is None or goal_points is None:
                print(f"Unable to compute iou statistics for run: {run_id}")
                continue
            curr_bbox = points_to_bbox(curr_points)
            goal_bbox = points_to_bbox(goal_points)
            iou = compute_iou(curr_bbox, goal_bbox)
            run_iou.append(iou)
        if len(run_iou) > 0:
            iou_stats.append({
                "run": run_id,
                "direction": direction,
                "iou_mean": np.mean(run_iou),
                "iou_median": np.median(run_iou),
                "iou_std": np.std(run_iou),
                "iou_min": np.min(run_iou),
                "iou_max": np.max(run_iou),
            })

    return pd.DataFrame(err_norm_stats), pd.DataFrame(iou_stats)


def compute_error_statistics_for_time_criteria(
        parquet_df: pd.DataFrame,
        duration_df: pd.DataFrame,
        threshold: float = 1.0,
        duration: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute error statistics for last 3 seconds and for first stable 3-second period.

    Returns:
        tuple: (err_norm_stats, iou_stats, stable_period_stats)
    """
    err_norm_stats = []
    iou_stats = []
    updated_duration_df = duration_df.copy()

    for run_id, run_group in parquet_df.groupby("run"):
        duration_row = duration_df[duration_df["run"] == run_id]
        if duration_row.empty:
            continue

        direction = duration_row.iloc[0]["direction"]
        original_flight_duration = duration_row.iloc[0]["flight_duration"]

        stable_period_data = get_first_stable_period(run_group, threshold=threshold, duration=duration)

        if stable_period_data is None or stable_period_data.empty:
            print(f"No stable period found for run: {run_id}")
            continue

        stable_period_end_time = stable_period_data["timestamp"].iloc[0]
        run_start_time = run_group["timestamp"].min()
        adjusted_flight_duration = stable_period_end_time - run_start_time + 3

        mask = updated_duration_df["run"] == run_id
        updated_duration_df.loc[mask, "flight_duration"] = adjusted_flight_duration

        err_uv_norms = stable_period_data["err_uv"].apply(lambda v: np.linalg.norm(v))
        err_norm_stats.append({
            "run": run_id,
            "direction": direction,
            "stable_period_start": stable_period_data["timestamp"].iloc[0],
            "stable_period_end": stable_period_data["timestamp"].iloc[-1],
            "stable_period_duration": stable_period_data["timestamp"].iloc[-1] - stable_period_data["timestamp"].iloc[0],
            "original_flight_duration": original_flight_duration,
            "adjusted_flight_duration": adjusted_flight_duration,
            "err_norm_mean": err_uv_norms.mean(),
            "err_norm_median": err_uv_norms.median(),
            "err_norm_std": err_uv_norms.std(),
            "err_norm_min": err_uv_norms.min(),
            "err_norm_max": err_uv_norms.max(),
        })

        run_iou = []
        for _, row in stable_period_data.iterrows():
            curr_points = row["current_points_flatten"]
            goal_points = row["goal_points_flatten"]
            if curr_points is None or goal_points is None:
                print(f"Unable to compute iou statistics for stable period in run: {run_id}")
                continue
            curr_bbox = points_to_bbox(curr_points)
            goal_bbox = points_to_bbox(goal_points)
            iou = compute_iou(curr_bbox, goal_bbox)
            run_iou.append(iou)

        if len(run_iou) > 0:
            iou_stats.append({
                "run": run_id,
                "direction": direction,
                "stable_period_start": stable_period_data["timestamp"].iloc[0],
                "stable_period_end": stable_period_data["timestamp"].iloc[-1],
                "stable_period_duration": stable_period_data["timestamp"].iloc[-1] -
                                          stable_period_data["timestamp"].iloc[0],
                "original_flight_duration": original_flight_duration,
                "adjusted_flight_duration": adjusted_flight_duration,
                "iou_mean": np.mean(run_iou),
                "iou_median": np.median(run_iou),
                "iou_std": np.std(run_iou),
                "iou_min": np.min(run_iou),
                "iou_max": np.max(run_iou),
            })

    return pd.DataFrame(err_norm_stats), pd.DataFrame(iou_stats), updated_duration_df

def get_first_stable_period(run_data: pd.DataFrame, threshold: float = 1.0, duration: float = 3.0) -> pd.DataFrame:
    """
    Get the DataFrame subset for the first period where |x_cmd|, |y_cmd|, and |rot_cmd|
    are all <= threshold for >= duration seconds.

    :param run_data: DataFrame containing the run data
    :param threshold: Maximum absolute value for commands to be considered stable
    :param duration: Minimum duration in seconds for a stability period
    :return: DataFrame subset of the first stable period, or None if no stable period found
    """
    if len(run_data) < 2:
        return None

    run_data = run_data.sort_values('timestamp').reset_index(drop=True)
    stable_mask = (
            (np.abs(run_data['x_cmd']) <= threshold) &
            (np.abs(run_data['y_cmd']) <= threshold) &
            (np.abs(run_data['rot_cmd']) <= threshold)
    )

    current_period_start_idx = None

    for idx, is_stable in enumerate(stable_mask):
        current_time = run_data.iloc[idx]['timestamp']

        if is_stable and current_period_start_idx is None:
            current_period_start_idx = idx
            current_period_start_time = current_time
        elif not is_stable and current_period_start_idx is not None:
            period_duration = current_time - current_period_start_time

            if period_duration >= duration:
                return run_data.iloc[current_period_start_idx:idx].copy()

            current_period_start_idx = None

    if current_period_start_idx is not None:
        final_time = run_data.iloc[-1]['timestamp']
        period_duration = final_time - current_period_start_time

        if period_duration >= duration:
            return run_data.iloc[current_period_start_idx:].copy()

    return None


def compute_flight_duration_distance_statistics(
        parquet_df: pd.DataFrame, duration_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    duration_stats = []
    distance_stats = []

    for run_id, run_group in parquet_df.groupby("run"):
        duration_row = duration_df[duration_df["run"] == run_id]
        if duration_row.empty:
            continue

        direction = duration_row.iloc[0]["direction"]
        flight_duration = duration_row.iloc[0]["flight_duration"]

        metadata_row = metadata_df[metadata_df["run"] == run_id]
        if not metadata_row.empty:
            distance = metadata_row.iloc[0]["distance"]
        else:
            distance = None
            print(f"No metadata found for run: {run_id}")

        duration_stats.append({
            "run": run_id,
            "direction": direction,
            "flight_duration": flight_duration,
        })

        if distance is not None:
            distance_stats.append({
                "run": run_id,
                "direction": direction,
                "distance": distance,
            })

    return pd.DataFrame(duration_stats), pd.DataFrame(distance_stats)


def compute_command_statistics(data_parquet: pd.DataFrame, duration_df: pd.DataFrame) -> pd.DataFrame:
    stats = []

    for run_id, run_group in data_parquet.groupby("run"):
        duration_row = duration_df[duration_df["run"] == run_id]
        if duration_row.empty:
            continue

        flight_end = duration_row.iloc[0]["end_time"]
        direction = duration_row.iloc[0]["direction"]
        time_cutoff = flight_end - 3

        run_last3s = run_group[run_group["timestamp"] >= time_cutoff].copy()
        if run_last3s.empty:
            continue

        run_last3s["x_cmd"] = run_last3s["x_cmd"].abs()
        run_last3s["y_cmd"] = run_last3s["y_cmd"].abs()
        run_last3s["rot_cmd"] = run_last3s["rot_cmd"].abs()

        stats.append({
            "run": run_id,
            "direction": direction,
            "x_cmd_mean": run_last3s["x_cmd"].mean(),
            "y_cmd_mean": run_last3s["y_cmd"].mean(),
            "rot_cmd_mean": run_last3s["rot_cmd"].mean(),
            "x_cmd_median": run_last3s["x_cmd"].median(),
            "y_cmd_median": run_last3s["y_cmd"].median(),
            "rot_cmd_median": run_last3s["rot_cmd"].median(),
            "x_cmd_std": run_last3s["x_cmd"].std(),
            "y_cmd_std": run_last3s["y_cmd"].std(),
            "rot_cmd_std": run_last3s["rot_cmd"].std(),
            "x_cmd_min": run_last3s["x_cmd"].min(),
            "y_cmd_min": run_last3s["y_cmd"].min(),
            "rot_cmd_min": run_last3s["rot_cmd"].min(),
            "x_cmd_max": run_last3s["x_cmd"].max(),
            "y_cmd_max": run_last3s["y_cmd"].max(),
            "rot_cmd_max": run_last3s["rot_cmd"].max(),
        })

    return pd.DataFrame(stats)


def plot_statistics_summary(err_stats: pd.DataFrame, iou_stats: pd.DataFrame, save_path_dir: Path, threshold: int = 0):
    def plot_metric(df, value_col, group_col, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df,
            x=group_col,
            y=value_col,
            hue="direction",
            # ci="sd",
            errorbar="sd",
            capsize=0.1
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(group_col.capitalize())
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path_dir / f"threshold_{threshold}-{filename}", dpi=300)
        plt.close()

    plot_metric(
        df=iou_stats,
        value_col="iou_mean",
        group_col="direction",
        title="Mean IoU per Direction at Last 3 Seconds",
        ylabel="IoU",
        filename="last_3_seconds_iou_mean_per_direction.png",
    )

    plot_metric(
        df=iou_stats,
        value_col="iou_median",
        group_col="direction",
        title="Median IoU per Direction at Last 3 Seconds",
        ylabel="IoU",
        filename="last_3_seconds_iou_median_per_direction.png",
    )

    plot_metric(
        df=err_stats,
        value_col="err_norm_mean",
        group_col="direction",
        title="Mean Error Norm per Direction at Last 3 Seconds",
        ylabel="L2 Error",
        filename="last_3_seconds_err_uv_mean_per_direction.png",
    )

    plot_metric(
        df=err_stats,
        value_col="err_norm_median",
        group_col="direction",
        title="Median Error Norm per Direction at Last 3 Seconds",
        ylabel="L2 Error",
        filename="last_3_seconds_err_uv_median_per_direction.png",
    )


def run_stats(
        base_path: str | Path,
        save_path_dir: str | Path,
        is_student: bool,
        is_real: bool,
):
    """
    Runs all the statistics evaluations for a given series of experiments.
    Will output per scene statistics and a mean per all scenes variants of files.
    The files generated are about command, distance, duration, errors and IoU statistics.
    Will also output the mean and median errors plots for the last 3 seconds of the mission.
    """
    base_path = Path(base_path)
    save_path_dir = Path(save_path_dir)
    save_path_name = os.path.basename(f"{save_path_dir}")
    save_path_dir.mkdir(parents=True, exist_ok=True)

    scenes_names = ConfigsDirName.REAL if is_real else ConfigsDirName.SIM

    print(f"Processing {base_path} - student: {is_student} - real: {is_real}")
    for index, scene_name in enumerate(scenes_names):
        if is_student:
            if is_real:
                scenes_names[index] = scene_name.replace("ibvs", "student")
            else:
                scenes_names[index] = f"{scene_name}-student"

    json_flight_data = load_json_flight_data(base_path, scenes_names)
    metadata_df = load_json_metadata(base_path, scenes_names)
    parquet_data = load_parquet_data(base_path, scenes_names, is_student=is_student)
    print(f"Total scenes: {parquet_data['scene'].nunique()}.")
    print(f"Total runs: {parquet_data['run'].nunique()}.")
    summary = parquet_data.copy().groupby('scene')['run'].nunique().reset_index()
    summary.columns = ['scene', 'unique_runs']
    summary = summary.sort_values('unique_runs', ascending=False)
    print(summary)

    threshold = 1
    err_stats, iou_stats = compute_error_statistics(parquet_data, json_flight_data)
    if is_student and not is_real:
        err_stats_1_thresh, iou_stats_1_thresh, updated_time_df = compute_error_statistics_for_time_criteria(
            parquet_data, json_flight_data, threshold=threshold, duration=3.0
        )
        err_stats_1_thresh.to_csv(save_path_dir / f"{save_path_name}_err_stats_1_threshold-total.csv", header=True)
        iou_stats_1_thresh.to_csv(save_path_dir / f"{save_path_name}_iou_stats_1_threshold-total.csv", header=True)
        plot_statistics_summary(err_stats_1_thresh, iou_stats_1_thresh, save_path_dir, threshold=threshold)
        err_stats_1_thresh = err_stats_1_thresh.groupby("direction").mean(numeric_only=True)
        iou_stats_1_thresh = iou_stats_1_thresh.groupby("direction").mean(numeric_only=True)
        err_stats_1_thresh.to_csv(save_path_dir / f"{save_path_name}_err_stats_threshold_{threshold}-mean.csv",
                                  header=True)
        iou_stats_1_thresh.to_csv(save_path_dir / f"{save_path_name}_iou_stats_threshold_{threshold}-mean.csv",
                                  header=True)

    cmd_stats = compute_command_statistics(parquet_data, json_flight_data)
    err_stats.to_csv(save_path_dir / f"{save_path_name}_error_stats-total.csv", header=True)
    iou_stats.to_csv(save_path_dir / f"{save_path_name}_iou_stats-total.csv", header=True)
    cmd_stats.to_csv(save_path_dir / f"{save_path_name}_cmd_stats-total.csv", header=True)

    plot_statistics_summary(err_stats, iou_stats, save_path_dir, threshold=0)

    err_stats_mean = err_stats.groupby("direction").mean(numeric_only=True)
    iou_stats_mean = iou_stats.groupby("direction").mean(numeric_only=True)
    cmd_stats_mean = cmd_stats.groupby("direction").mean(numeric_only=True)

    err_stats_mean.to_csv(save_path_dir / f"{save_path_name}_error_stats-mean.csv", header=True)
    iou_stats_mean.to_csv(save_path_dir / f"{save_path_name}_iou_stats-mean.csv", header=True)
    cmd_stats_mean.to_csv(save_path_dir / f"{save_path_name}_cmd_stats-mean.csv", header=True)
    duration_stats, distance_stats = compute_flight_duration_distance_statistics(
        parquet_data, json_flight_data, metadata_df
    )
    duration_stats.to_csv(save_path_dir / f"{save_path_name}_duration_stats-total.csv", header=True)
    distance_stats.to_csv(save_path_dir / f"{save_path_name}_distance_stats-total.csv", header=True)

    duration_stats_mean = duration_stats.groupby("direction").mean(numeric_only=True)
    duration_stats_median = duration_stats.groupby("direction").median(numeric_only=True)
    distance_stats_mean = distance_stats.groupby("direction").mean(numeric_only=True)
    distance_stats_median = distance_stats.groupby("direction").median(numeric_only=True)
    duration_stats_mean_suffixed = duration_stats_mean.add_suffix('_mean')
    duration_stats_median_suffixed = duration_stats_median.add_suffix('_median')
    duration_stats_combined = pd.concat([duration_stats_mean_suffixed, duration_stats_median_suffixed], axis=1)
    distance_stats_mean_suffixed = distance_stats_mean.add_suffix('_mean')
    distance_stats_median_suffixed = distance_stats_median.add_suffix('_median')
    distance_stats_combined = pd.concat([distance_stats_mean_suffixed, distance_stats_median_suffixed], axis=1)
    duration_stats_combined.to_csv(save_path_dir / f"{save_path_name}_duration_stats-mean_median.csv", header=True)
    distance_stats_combined.to_csv(save_path_dir / f"{save_path_name}_distance_stats-mean_median.csv", header=True)

    print("Error Stats:")
    print(err_stats_mean)
    print("IoU Stats:")
    print(iou_stats_mean)
    print("Command Stats:")
    print(cmd_stats_mean)
    print("Duration Stats:")
    print(duration_stats_combined)
    print("Distance Stats:")
    print(distance_stats_combined)


if __name__ == '__main__':
    from auto_follow.utils.path_manager import Paths

    pd.set_option('display.max_columns', None)

    save_path = Paths.BASE_DIR / "results" / "statistics"

    sim_ibvs_path = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/ibvs/sim-ibvs-results-merged")
    run_stats(
        base_path=sim_ibvs_path,
        save_path_dir=save_path / "sim-ibvs-results",
        is_student=False,
        is_real=False,
    )

    sim_student_path = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student-with-teacher-output/sim-student-with-teacher-output-merged")
    run_stats(
        base_path=sim_student_path,
        save_path_dir=save_path / "sim-student-results",
        is_student=True,
        is_real=False,
    )

    real_ibvs_path = Path(
        "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged")
    run_stats(
        base_path=real_ibvs_path,
        save_path_dir=save_path / "real-ibvs-results",
        is_student=False,
        is_real=True,
    )

    real_student_path = Path(
        "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/student/real-student-with-teacher-output-new/results-real-student-21-06")
    run_stats(
        base_path=real_student_path,
        save_path_dir=save_path / "real-student-results",
        is_student=True,
        is_real=True,
    )
