from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from auto_follow.visualization.load_data import load_json_flight_data, load_parquet_data, load_json_metadata, \
    ConfigsDirName


def plot_flight_durations(durations_df, save_path: str | Path = Path("./plot-output")):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=durations_df, x="run", y="flight_duration", hue="scene")
    plt.title("Flight Duration per Run")
    plt.xticks(rotation=45, ha="right")
    plt.xticks([], [])
    plt.savefig(Path(save_path) / "flight_durations_for_each_run_barplot.png")
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=durations_df, x="run", y="flight_duration", hue="scene", marker="o", linewidth=1)
    plt.xticks([], [])
    plt.title("Flight Duration per Run (Line Plot)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(save_path) / "flight_durations_for_each_run_lineplot.png")
    plt.close()

    plt.figure()
    sns.barplot(data=durations_df, x="direction", y="flight_duration", hue="direction")
    plt.title("Flight Duration per Run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(Path(save_path) / "flight_durations.png")
    plt.close()

    plt.figure()
    sns.boxplot(data=durations_df, x="direction", y="flight_duration", )
    plt.title("Flight Duration Distribution by Direction")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(Path(save_path) / "flight_durations_by_direction.png")
    plt.close()

    plt.figure()
    sns.histplot(durations_df["flight_duration"], bins=20, kde=True)
    plt.title("Flight Duration Distribution")
    plt.savefig(Path(save_path) / "flight_durations_histogram.png")
    plt.close()


def plot_distances(distances_df, save_path: str | Path = Path("./plot-output")):
    """
    Create distance plots similar to flight duration plots
    """
    save_path = Path(save_path)

    # Bar plot by run
    plt.figure(figsize=(10, 6))
    sns.barplot(data=distances_df, x="run", y="distance", hue="scene")
    plt.title("Distance per Run")
    plt.xticks(rotation=45, ha="right")
    plt.xticks([], [])
    plt.tight_layout()
    plt.savefig(save_path / "distances_for_each_run_barplot.png")
    plt.close()

    # Line plot by run
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=distances_df, x="run", y="distance", hue="scene", marker="o", linewidth=1)
    plt.xticks([], [])
    plt.title("Distance per Run (Line Plot)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / "distances_for_each_run_lineplot.png")
    plt.close()

    # Bar plot by direction
    sns.barplot(data=distances_df, x="direction", y="distance", hue="direction")
    plt.title("Distance per Direction")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path / "distances_by_direction.png")
    plt.close()

    # Box plot by direction
    sns.boxplot(data=distances_df, x="direction", y="distance")
    plt.title("Distance Distribution by Direction")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path / "distances_by_direction_boxplot.png")
    plt.close()

    # Histogram
    sns.histplot(distances_df["distance"], bins=20, kde=True)
    plt.title("Distance Distribution")
    plt.tight_layout()
    plt.savefig(save_path / "distances_histogram.png")
    plt.close()


def plot_errors(parquet_df: pd.DataFrame, save_path: str | Path = Path("./plot-output")):
    parquet_df["err_norm"] = parquet_df["err_uv"].apply(lambda x: np.linalg.norm(x))

    plt.figure(figsize=(12, 6))
    sns.histplot(parquet_df["err_norm"], bins=50, kde=True)
    plt.title("Tracking Error Norm Over All Runs")
    plt.tight_layout()
    plt.savefig(save_path / "error_norm_runs.png")
    plt.close()


def plot_random_run(parquet_df: pd.DataFrame, save_path: str | Path = Path("./plot-output")):
    random_run_id = parquet_df["run"].sample(1).iloc[0]
    sample_run = parquet_df[parquet_df["run"] == random_run_id].copy()
    sample_run["err_norm"] = sample_run["err_uv"].apply(lambda x: np.linalg.norm(x))

    sample_run["Time"] = sample_run["timestamp"] - sample_run["timestamp"].iloc[0]

    fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    sns.lineplot(data=sample_run, x="Time", y="x_cmd", ax=ax[0])
    sns.lineplot(data=sample_run, x="Time", y="y_cmd", ax=ax[1])
    sns.lineplot(data=sample_run, x="Time", y="rot_cmd", ax=ax[2])
    sns.lineplot(data=sample_run, x="Time", y="err_norm", ax=ax[3])
    for a in ax:
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.suptitle(f"Run {random_run_id} on scene {sample_run['scene'].values[0]}", fontsize=16)
    plt.savefig(save_path / f"time_series_sample_run_{random_run_id}.png")
    plt.close()


def plot_command_distribution(parquet_df: pd.DataFrame, save_path: str | Path = Path("./plot-output")):
    x_min, x_max = parquet_df["x_cmd"].min(), parquet_df["x_cmd"].max()
    y_min, y_max = parquet_df["y_cmd"].min(), parquet_df["y_cmd"].max()
    rot_min, rot_max = parquet_df["rot_cmd"].min(), parquet_df["rot_cmd"].max()

    parquet_df["x_cmd_norm"] = 2 * (parquet_df["x_cmd"] - x_min) / (x_max - x_min) - 1
    parquet_df["y_cmd_norm"] = 2 * (parquet_df["y_cmd"] - y_min) / (y_max - y_min) - 1
    parquet_df["rot_cmd_norm"] = 2 * (parquet_df["rot_cmd"] - rot_min) / (rot_max - rot_min) - 1

    # Set number of bins (e.g., 50) for each command
    num_bins = 50
    x_bins = np.linspace(x_min, x_max, num_bins + 1)
    y_bins = np.linspace(y_min, y_max, num_bins + 1)
    rot_bins = np.linspace(rot_min, rot_max, num_bins + 1)

    plt.figure(figsize=(15, 5))

    # X_cmd distribution
    plt.subplot(1, 3, 1)
    sns.histplot(data=parquet_df, x="x_cmd", kde=True, bins=x_bins)
    plt.title("Distribution of x_cmd")
    plt.xlabel("x_cmd")
    plt.ylabel("Count")

    # Y_cmd distribution
    plt.subplot(1, 3, 2)
    sns.histplot(data=parquet_df, x="y_cmd", kde=True, bins=y_bins)
    plt.title("Distribution of y_cmd")
    plt.xlabel("y_cmd")
    plt.ylabel("Count")

    # Rot_cmd distribution
    plt.subplot(1, 3, 3)
    sns.histplot(data=parquet_df, x="rot_cmd", kde=True, bins=rot_bins)
    plt.title("Distribution of rot_cmd")
    plt.xlabel("rot_cmd")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path / "command_distributions.png")

    # Print summary statistics for both original and normalized columns
    print("\nSummary Statistics (Original):")
    print(parquet_df[["x_cmd", "y_cmd", "rot_cmd"]].describe())
    print("\nSummary Statistics (Normalized):")
    print(parquet_df[["x_cmd_norm", "y_cmd_norm", "rot_cmd_norm"]].describe())


def clean_config_name(config):
    return (
        config
        .replace("bunker-online-4k-config-test-", "")
        .replace("real-ibvs-", "")
        .replace("real-student-", "")
        .replace("-student", "")
    )


def plot_velocity_distributions(
        parquet_df: pd.DataFrame,
        config_dirs: list[str],
        save_path: str | Path = Path("./plot-output")
):
    err_min = parquet_df["err_norm"].min()
    err_max = parquet_df["err_norm"].max()
    if err_max > err_min:
        parquet_df["err_norm"] = (parquet_df["err_norm"] - err_min) / (err_max - err_min)
    else:
        parquet_df["err_norm"] = 0.5

    # Normalize timestamps within each run to percentage (0-100)
    logs_df_normalized = parquet_df.copy()
    for run_id in parquet_df["run"].unique():
        run_mask = logs_df_normalized["run"] == run_id
        run_timestamps = logs_df_normalized.loc[run_mask, "timestamp"]
        if len(run_timestamps) > 1:
            min_time = run_timestamps.min()
            max_time = run_timestamps.max()
            if max_time > min_time:
                normalized_time = 100 * (run_timestamps - min_time) / (max_time - min_time)
                logs_df_normalized.loc[run_mask, "normalized_time"] = normalized_time
            else:
                logs_df_normalized.loc[run_mask, "normalized_time"] = 0
        else:
            logs_df_normalized.loc[run_mask, "normalized_time"] = 0

    # Create time bins for aggregation
    time_bins = np.linspace(0, 100, 21)  # 20 bins from 0% to 100%
    logs_df_normalized["time_bin"] = pd.cut(logs_df_normalized["normalized_time"],
                                            bins=time_bins, labels=False, include_lowest=True)
    config_label_map = {cfg: clean_config_name(cfg) for cfg in config_dirs}

    # Calculate statistics for each config and time bin
    stats_by_config = []
    for config in config_dirs:
        config_data = logs_df_normalized[logs_df_normalized["scene"] == config]
        if len(config_data) == 0:
            continue

        for bin_idx in range(20):  # 20 bins
            bin_data = config_data[config_data["time_bin"] == bin_idx]
            if len(bin_data) > 0:
                time_center = (time_bins[bin_idx] + time_bins[bin_idx + 1]) / 2
                stats_by_config.append({
                    "scene": config,
                    "time_percent": time_center,
                    "x_cmd_mean": bin_data["x_cmd"].mean(),
                    "x_cmd_std": bin_data["x_cmd"].std(),
                    "y_cmd_mean": bin_data["y_cmd"].mean(),
                    "y_cmd_std": bin_data["y_cmd"].std(),
                    "rot_cmd_mean": bin_data["rot_cmd"].mean(),
                    "rot_cmd_std": bin_data["rot_cmd"].std(),
                    "err_norm_mean": bin_data["err_norm"].mean(),
                    "err_norm_std": bin_data["err_norm"].std(),
                    "sample_count": len(bin_data)
                })

    stats_df = pd.DataFrame(stats_by_config)

    # Create comprehensive time series plots
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle("Command and Error Statistics Over Time by Configuration", fontsize=16, fontweight="bold")

    commands = ["x_cmd", "y_cmd", "rot_cmd", "err_norm"]
    command_labels = ["X Command", "Y Command", "Rotation Command", "Error Norm"]

    colors = plt.cm.Set3(np.linspace(0, 1, len(config_dirs)))

    for i, (cmd, label) in enumerate(zip(commands, command_labels)):
        # Mean plot
        ax_mean = axes[i, 0]
        for j, config in enumerate(config_dirs):
            config_stats = stats_df[stats_df["scene"] == config]
            if config_stats.empty:
                continue
            config_label = config_label_map.get(config, config)
            ax_mean.plot(config_stats["time_percent"], config_stats[f"{cmd}_mean"],
                         color=colors[j], marker="o", markersize=3, linewidth=2,
                         label=config_label, alpha=0.8)

        ax_mean.set_title(f"{label} - Mean Over Time", fontsize=12, fontweight="bold")
        ax_mean.set_xlabel("Flight Progress (%)", fontsize=10)
        ax_mean.set_ylabel(f"Mean {label}", fontsize=10)
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # Std plot
        ax_std = axes[i, 1]
        for j, config in enumerate(config_dirs):
            config_stats = stats_df[stats_df["scene"] == config]
            if len(config_stats) > 0:
                config_label = config_label_map.get(config, config)
                ax_std.plot(config_stats["time_percent"], config_stats[f"{cmd}_std"],
                            color=colors[j], marker="s", markersize=3, linewidth=2,
                            label=config_label, alpha=0.8)

        ax_std.set_title(f"{label} - Standard Deviation Over Time", fontsize=12, fontweight="bold")
        ax_std.set_xlabel("Flight Progress (%)", fontsize=10)
        ax_std.set_ylabel(f"Std Dev {label}", fontsize=10)
        ax_std.grid(True, alpha=0.3)
        ax_std.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path / "evolution_over_time_commands_error_stats.png", dpi=300, bbox_inches="tight")
    plt.close()

    fine_time_bins = np.linspace(0, 100, 51)  # 50 bins for smoother curves
    logs_df_normalized["fine_time_bin"] = pd.cut(logs_df_normalized["normalized_time"],
                                                 bins=fine_time_bins, labels=False, include_lowest=True)

    # Calculate statistics for each config and fine time bin
    fine_stats_by_config = []
    for config in config_dirs:
        config_data = logs_df_normalized[logs_df_normalized["scene"] == config]
        if len(config_data) == 0:
            continue

        for bin_idx in range(50):  # 50 bins
            bin_data = config_data[config_data["fine_time_bin"] == bin_idx]
            if len(bin_data) > 0:
                time_center = (fine_time_bins[bin_idx] + fine_time_bins[bin_idx + 1]) / 2

                # Calculate percentiles for confidence intervals
                x_cmd_values = bin_data["x_cmd"].dropna()
                y_cmd_values = bin_data["y_cmd"].dropna()
                rot_cmd_values = bin_data["rot_cmd"].dropna()
                err_norm_values = bin_data["err_norm"].dropna()

                fine_stats_by_config.append({
                    "scene": config,
                    "time_percent": time_center,
                    "x_cmd_mean": x_cmd_values.mean() if len(x_cmd_values) > 0 else np.nan,
                    "x_cmd_std": x_cmd_values.std() if len(x_cmd_values) > 0 else np.nan,
                    "x_cmd_25": x_cmd_values.quantile(0.25) if len(x_cmd_values) > 0 else np.nan,
                    "x_cmd_75": x_cmd_values.quantile(0.75) if len(x_cmd_values) > 0 else np.nan,
                    "y_cmd_mean": y_cmd_values.mean() if len(y_cmd_values) > 0 else np.nan,
                    "y_cmd_std": y_cmd_values.std() if len(y_cmd_values) > 0 else np.nan,
                    "y_cmd_25": y_cmd_values.quantile(0.25) if len(y_cmd_values) > 0 else np.nan,
                    "y_cmd_75": y_cmd_values.quantile(0.75) if len(y_cmd_values) > 0 else np.nan,
                    "rot_cmd_mean": rot_cmd_values.mean() if len(rot_cmd_values) > 0 else np.nan,
                    "rot_cmd_std": rot_cmd_values.std() if len(rot_cmd_values) > 0 else np.nan,
                    "rot_cmd_25": rot_cmd_values.quantile(0.25) if len(rot_cmd_values) > 0 else np.nan,
                    "rot_cmd_75": rot_cmd_values.quantile(0.75) if len(rot_cmd_values) > 0 else np.nan,
                    "err_norm_mean": err_norm_values.mean() if len(err_norm_values) > 0 else np.nan,
                    "err_norm_std": err_norm_values.std() if len(err_norm_values) > 0 else np.nan,
                    "err_norm_25": err_norm_values.quantile(0.25) if len(err_norm_values) > 0 else np.nan,
                    "err_norm_75": err_norm_values.quantile(0.75) if len(err_norm_values) > 0 else np.nan,
                    "sample_count": len(bin_data)
                })

    fine_stats_df = pd.DataFrame(fine_stats_by_config)

    # Create RL-style plots for each command
    commands_to_plot = [
        ("x_cmd", "X Command Evolution", "X Command Value"),
        ("y_cmd", "Y Command Evolution", "Y Command Value"),
        ("rot_cmd", "Rotation Command Evolution", "Rotation Command Value"),
        ("err_norm", "Error Evolution", "Error Norm")
    ]

    # Define a nice color palette
    config_colors = {
        "down-left": "#1f77b4",
        "down-right": "#ff7f0e",
        "left": "#2ca02c",
        "right": "#d62728",
        "up-left": "#9467bd",
        "up-right": "#8c564b",
        "front-small-offset-right": "#e377c2",
        "front-small-offset-left": "#7f7f7f"
    }

    for cmd, title, ylabel in commands_to_plot:
        plt.figure(figsize=(12, 8))

        for config in config_dirs:
            config_stats = fine_stats_df[fine_stats_df["scene"] == config].dropna(subset=[f"{cmd}_mean"])
            if len(config_stats) == 0:
                continue

            config_label = config_label_map.get(config, config)

            color = config_colors.get(config_label, "#000000")

            # Get data
            x = config_stats["time_percent"].values
            y_mean = config_stats[f"{cmd}_mean"].values
            y_std = config_stats[f"{cmd}_std"].values
            y_25 = config_stats[f"{cmd}_25"].values
            y_75 = config_stats[f"{cmd}_75"].values

            # Plot mean line
            plt.plot(x, y_mean, color=color, linewidth=2.5, label=config_label, alpha=0.9)

            # Plot confidence interval using standard deviation
            plt.fill_between(x, y_mean - y_std, y_mean + y_std,
                             color=color, alpha=0.2)

            # Alternative: use percentiles for confidence interval (uncomment if preferred)
            # plt.fill_between(x, y_25, y_75, color=color, alpha=0.2)

        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Flight Progress (%)", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        plt.legend(fontsize=12, loc="best")

        # Set axis limits
        if cmd in ["x_cmd", "y_cmd", "rot_cmd"]:
            plt.ylim(-25, 25)
        elif cmd == "err_norm":
            plt.ylim(0, 1)

        # Style similar to RL plot
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_linewidth(1)
        plt.gca().spines["bottom"].set_linewidth(1)

        plt.tight_layout()
        plt.savefig(save_path / f"evolution_over_time_{cmd}_rl_style.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Create a combined plot similar to the RL style showing all commands
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]

    for i, (cmd, title, ylabel) in enumerate(commands_to_plot):
        ax = axes[i]

        for config in config_dirs:
            config_stats = fine_stats_df[fine_stats_df["scene"] == config].dropna(subset=[f"{cmd}_mean"])
            if len(config_stats) == 0:
                continue

            config_label = config_label_map.get(config, config)
            color = config_colors.get(config_label, "#000000")

            # Get data
            x = config_stats["time_percent"].values
            y_mean = config_stats[f"{cmd}_mean"].values
            y_std = config_stats[f"{cmd}_std"].values

            # Plot mean line
            ax.plot(x, y_mean, color=color, linewidth=2, label=config_label, alpha=0.9)

            # Plot confidence interval
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                            color=color, alpha=0.2)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Flight Progress (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        # Set axis limits
        if cmd in ["x_cmd", "y_cmd", "rot_cmd"]:
            ax.set_ylim(-25, 25)
        elif cmd == "err_norm":
            ax.set_ylim(0, 1)

        # Style similar to RL plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

        if i == 0:  # Only show legend on first subplot
            ax.legend(fontsize=10, loc="best")

    plt.suptitle("Command and Error Evolution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "evolution_over_time_commands_errors_rl_style_combined.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_plot_analysis_on_scenes(
        base_path: str | Path,
        save_path_name: str,
        scenes_names: list[str],
        is_student: bool,
        is_real: bool,
        random_runs: int = 1
):
    reset_matplotlib_state()

    base_path = Path(base_path)
    save_path_dir = Path(f"./{save_path_name}")
    save_path_dir.mkdir(parents=True, exist_ok=True)

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

    summary = parquet_data.groupby('scene')['run'].nunique().reset_index()
    summary.columns = ['scene', 'unique_runs']
    summary = summary.sort_values('unique_runs', ascending=False)
    print(summary)

    plot_flight_durations(json_flight_data, save_path=save_path_dir)
    plot_distances(metadata_df, save_path=save_path_dir)
    plot_command_distribution(parquet_data, save_path=save_path_dir)
    plot_errors(parquet_data, save_path=save_path_dir)
    plot_velocity_distributions(parquet_data, config_dirs=scenes_names, save_path=save_path_dir)
    for _ in range(random_runs):
        plot_random_run(parquet_data, save_path=save_path_dir)


def reset_matplotlib_state():
    """Reset matplotlib to clean state between plotting sessions"""
    plt.clf()
    plt.cla()
    plt.close('all')
    plt.rcdefaults()
    sns.reset_defaults()
    # sns.set_theme()


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    sim_ibvs_path = "/home/brittle/Desktop/work/Data/car-data/droid-data/sim/sim-ibvs-results-merged"
    sim_ibvs_path = Path(sim_ibvs_path)
    try:
        run_plot_analysis_on_scenes(
            base_path=sim_ibvs_path,
            save_path_name="plots-sim-ibvs",
            scenes_names=ConfigsDirName.SIM,
            is_student=False,
            is_real=False,
            random_runs=5
        )
    except Exception as e:
        print(f"SIM prob: {e}")

    # sim_student_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student-with-teacher-output/sim-student-with-teacher-output-merged"
    # sim_student_path = Path(sim_student_path)
    # try:
    #     run_plot_analysis_on_scenes(
    #         base_path=sim_student_path,
    #         save_path_name="plots-sim-student",
    #         scenes_names=ConfigsDirName.SIM,
    #         is_student=True,
    #         is_real=False,
    #         random_runs=5
    #     )
    # except Exception as e:
    #     print(f"SIM prob: {e}")
    #
    # real_ibvs_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged"
    # real_ibvs_path = Path(real_ibvs_path)
    # try:
    #     run_plot_analysis_on_scenes(
    #         base_path=real_ibvs_path,
    #         save_path_name="plots-real-ibvs",
    #         scenes_names=ConfigsDirName.REAL,
    #         is_student=False,
    #         is_real=True,
    #         random_runs=5
    #     )
    # except Exception as e:
    #     print(f"SIM prob: {e}")
    #
    # real_student_path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/student/real-student-with-teacher-output"
    # real_student_path = Path(real_student_path)
    # try:
    #     run_plot_analysis_on_scenes(
    #         base_path=real_student_path,
    #         save_path_name="plots-real-student",
    #         scenes_names=ConfigsDirName.REAL,
    #         is_student=True,
    #         is_real=True,
    #         random_runs=5
    #     )
    # except Exception as e:
    #     print(f"SIM prob: {e}")
