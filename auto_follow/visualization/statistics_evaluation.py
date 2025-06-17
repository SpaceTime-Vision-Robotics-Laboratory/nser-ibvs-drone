from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_error_statistics(data_parquet: pd.DataFrame, duration_df: pd.DataFrame) -> pd.DataFrame:
    stats = []

    for run_id, run_group in data_parquet.groupby("run"):
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
        stats.append({
            "run": run_id,
            "direction": direction,
            "err_uv_mean": err_uv_norms.mean(),
            "err_uv_median": err_uv_norms.median(),
            "err_uv_std": err_uv_norms.std(),
            "err_uv_min": err_uv_norms.min(),
            "err_uv_max": err_uv_norms.max(),
        })

    stats = pd.DataFrame(stats)

    stats_to_print = stats.groupby("direction").mean(numeric_only=True)
    stats_to_print.to_csv("temp_table.csv", index=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=stats_to_print.values,
                     colLabels=stats_to_print.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.savefig('table.png', bbox_inches='tight', dpi=300)
    plt.show()

    return pd.DataFrame(stats)


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


if __name__ == '__main__':
    from auto_follow.visualization.load_data import load_json_flight_data, load_parquet_data

    BASE_DIR = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/tests-merged")
    BASE_DIR = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/ibvs/sim-ibvs-results-merged")
    BASE_DIR = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student/sim-student-results-merged")

    pd.set_option('display.max_columns', None)

    BASE_CONFIGS = [
        "bunker-online-4k-config-test-down-left",
        "bunker-online-4k-config-test-down-right",
        "bunker-online-4k-config-test-left",
        "bunker-online-4k-config-test-right",
        "bunker-online-4k-config-test-up-left",
        "bunker-online-4k-config-test-up-right",
        "bunker-online-4k-config-test-front-small-offset-right",
        "bunker-online-4k-config-test-front-small-offset-left"
    ]
    IS_STUDENT = True
    BASE_CONFIGS = [
        f"{name}-student" if IS_STUDENT else name
        for name in BASE_CONFIGS
    ]

    BASE_DIR = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged")
    BASE_CONFIGS = [
        "real-ibvs-down-left",
        "real-ibvs-front-small-offset-left",
        "real-ibvs-left",
        "real-ibvs-up-left",
        "real-ibvs-down-right",
        "real-ibvs-front-small-offset-right",
        "real-ibvs-right",
        "real-ibvs-up-right",
    ]

    json_flight_data = load_json_flight_data(BASE_DIR, BASE_CONFIGS)
    parquet_data = load_parquet_data(BASE_DIR, BASE_CONFIGS)

    err_stats = compute_error_statistics(parquet_data, json_flight_data)
    cmd_stats = compute_command_statistics(parquet_data, json_flight_data)

    print(err_stats.groupby("direction").mean(numeric_only=True))
    print(cmd_stats.groupby("direction").mean(numeric_only=True))
