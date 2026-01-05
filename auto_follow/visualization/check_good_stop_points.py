import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import math

# sns.set(style="whitegrid")

BASE_DIR = "/home/brittle/Desktop/work/code/space-time-lab-org/auto-follow/output/results_without_frames_poli_pc"
BASE_DIR = "/home/brittle/Desktop/work/Data/car-ibvs-data/real_world_results"
BASE_DIR = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real-world-ibvs-results"
# BASE_DIR = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/car-ibvs-sim-results-good"

SAVE_PATH = Path("./plot-output")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_DIRS = [
    "bunker-online-4k-config-test-down-left",
    "bunker-online-4k-config-test-down-right",

    "bunker-online-4k-config-test-left",
    "bunker-online-4k-config-test-right",

    "bunker-online-4k-config-test-up-left",
    "bunker-online-4k-config-test-up-right",

    "bunker-online-4k-config-test-front-small-offset-right",
    "bunker-online-4k-config-test-front-small-offset-left"
]

CONFIG_DIRS = [
    "real-ibvs-down-left",
    "real-ibvs-front-small-offset-left",
    "real-ibvs-left",
    "real-ibvs-up-left",
    "real-ibvs-down-right",
    "real-ibvs-front-small-offset-right",
    "real-ibvs-right",
    "real-ibvs-up-right",
]
is_student = False
if is_student:
    CONFIG_DIRS = [f"{directory_name}-student" for directory_name in CONFIG_DIRS]


def load_flight_durations():
    all_data = []
    for config in CONFIG_DIRS:
        folder = os.path.join(BASE_DIR, config, "results")
        if not os.path.exists(folder):
            continue

        for run in os.listdir(folder):
            path = os.path.join(folder, run, "flight_duration.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    data["run"] = run
                    data["config"] = config

                    if "down-left" in config:
                        data["direction"] = "down-left"
                    elif "down-right" in config:
                        data["direction"] = "down-right"
                    elif "up-left" in config:
                        data["direction"] = "up-left"
                    elif "up-right" in config:
                        data["direction"] = "up-right"
                    elif config.endswith("left"):
                        data["direction"] = "left"
                    elif config.endswith("right"):
                        data["direction"] = "right"
                    else:
                        data["direction"] = "unknown"

                    all_data.append(data)
    return pd.DataFrame(all_data)


def load_parquet_logs():
    all_logs = []
    for config in CONFIG_DIRS:
        print(config)
        folder = os.path.join(BASE_DIR, config, "parquet-logs")
        if not os.path.exists(folder):
            continue

        for run in os.listdir(folder):
            parquet_file = os.path.join(folder, run, "logs.parquet")
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                df["run"] = run
                df["config"] = config

                if "down-left" in config:
                    df["direction"] = "down-left"
                elif "down-right" in config:
                    df["direction"] = "down-right"
                elif "up-left" in config:
                    df["direction"] = "up-left"
                elif "up-right" in config:
                    df["direction"] = "up-right"
                elif config.endswith("left"):
                    df["direction"] = "left"
                elif config.endswith("right"):
                    df["direction"] = "right"
                else:
                    df["direction"] = "unknown"

                all_logs.append(df)
    return pd.concat(all_logs, ignore_index=True)


def extract_vector_norms(df):
    """Extract norms from vector columns"""
    if 'err_uv' in df.columns:
        df['err_norm'] = df['err_uv'].apply(
            lambda x: np.linalg.norm(x) if isinstance(x, list | np.ndarray) else np.nan
        )

    if 'velocity' in df.columns:
        df['velocity_norm'] = df['velocity'].apply(
            lambda x: np.linalg.norm(x) if isinstance(x, list | np.ndarray) else np.nan
        )

    command_cols = ['x_cmd', 'y_cmd', 'z_cmd', 'rot_cmd']
    available_cmd_cols = [col for col in command_cols if col in df.columns]
    if available_cmd_cols:
        df['cmd_norm'] = df[available_cmd_cols].apply(lambda row: np.linalg.norm(row.values), axis=1)

    return df


def main():
    print("Loading parquet logs...")
    logs_df = load_parquet_logs()
    print(logs_df.head())

    # Error vector norm (err_uv) - normalized between 0 and 1
    logs_df["err_norm"] = logs_df["err_uv"].apply(lambda x: np.linalg.norm(x))
    print(f"Columns: {logs_df.columns}")

    # # Jacobian condition number at the end of runs
    # print("Extracting final jcond values for each run...")
    # final_errors = []
    # for run_id in logs_df['run'].unique():
    #     run_data = logs_df[logs_df['run'] == run_id]
    #     if len(run_data) > 0:
    #         # Get the last 50 jcond values for this run (or all available if less than 50)
    #         last_jcond_values = run_data['jcond'].iloc[-15:]
    #         final_jcond = last_jcond_values.median()
    #         final_jcond_values.append({
    #             'run': run_id,
    #             'config': run_data['config'].iloc[0],
    #             'direction': run_data['direction'].iloc[0],
    #             'final_jcond': final_jcond
    #         })

    error_norms = logs_df[logs_df['err_norm'] <= 40]
    error_norms["rot_cmd"] = [int(compute_rotation(val)) for val in error_norms["velocity"].tolist()]
    print(f"Error Norms: {error_norms.head}")
    print(error_norms.head)
    print(
        f"Max -> X: {np.max(error_norms['x_cmd']):4d} | Y: {np.max(error_norms['y_cmd']):4d} | R: {np.max(error_norms['rot_cmd']):4d}")
    print(
        f"Med -> X: {np.median(error_norms['x_cmd']):.4f} | Y: {np.median(error_norms['y_cmd']):.4f} | R: {np.median(error_norms['rot_cmd']):.4f}")
    print(
        f"Man -> X: {np.mean(error_norms['x_cmd']):.4f} | Y: {np.mean(error_norms['y_cmd']):.4f} | R: {np.mean(error_norms['rot_cmd']):.4f}")
    print(
        f"STD -> X: {np.std(error_norms['x_cmd']):.4f} | Y: {np.std(error_norms['y_cmd']):.4f} | R: {np.std(error_norms['rot_cmd']):.4f}")
    print(len(logs_df["err_norm"]))


def compute_rotation(values):
    return math.ceil(100 * values[2] / np.deg2rad(60))


if __name__ == "__main__":
    main()
