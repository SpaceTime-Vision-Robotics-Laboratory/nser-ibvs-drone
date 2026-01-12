import json
import os
from pathlib import Path

import pandas as pd

from nser_ibvs_drone.visualization.metadata_trajectory_analysis import calculate_drone_distance


class ConfigsDirName:
    SIM = [
        "bunker-online-4k-config-test-down-left",
        "bunker-online-4k-config-test-down-right",
        "bunker-online-4k-config-test-left",
        "bunker-online-4k-config-test-right",
        "bunker-online-4k-config-test-up-left",
        "bunker-online-4k-config-test-up-right",
        "bunker-online-4k-config-test-front-small-offset-right",
        "bunker-online-4k-config-test-front-small-offset-left"
    ]

    REAL = [
        "real-ibvs-down-left",
        "real-ibvs-front-small-offset-left",
        "real-ibvs-left",
        "real-ibvs-up-left",
        "real-ibvs-down-right",
        "real-ibvs-front-small-offset-right",
        "real-ibvs-right",
        "real-ibvs-up-right",
    ]

    SIM_STUDENT = [f"{directory_name}-student" for directory_name in SIM]
    REAL_STUDENT = [f"{directory_name}-student" for directory_name in REAL]


def extract_direction(config_name: str) -> str:
    direction_keywords = [
        "front-small-offset-right",
        "front-small-offset-left",
        "down-left",
        "down-right",
        "up-left",
        "up-right",
        "left",
        "right"
    ]
    for keyword in direction_keywords:
        if keyword in config_name:
            return keyword
    return "unknown"


def load_json_flight_data(results_dir: str | Path, scene_dir_names: list[str]) -> pd.DataFrame:
    all_data = []
    results_dir = Path(results_dir)
    for scene_name in scene_dir_names:
        results_path = results_dir / scene_name / "results"
        if not results_path.exists():
            print(f"No results directory for {scene_name} check the path ({results_path})")
            continue
        scene_direction = extract_direction(scene_name)

        for experiment_run in os.listdir(results_path):
            duration_file = results_path / experiment_run / "flight_duration.json"
            if duration_file.exists():
                with open(duration_file) as f:
                    data = json.load(f)
                data.update({
                    "run": experiment_run,
                    "scene": scene_name,
                    "direction": scene_direction,
                })
                all_data.append(data)
    return pd.DataFrame(all_data)


def load_json_metadata(results_dir: str | Path, scene_dir_names: list[str]) -> pd.DataFrame:
    all_metadata = []
    results_dir = Path(results_dir)

    for scene_name in scene_dir_names:
        results_path = results_dir / scene_name / "results"
        if not results_path.exists():
            print(f"No results directory for {scene_name} check the path ({results_path})")
            continue

        scene_direction = extract_direction(scene_name)

        for experiment_run in os.listdir(results_path):
            metadata_file = results_path / experiment_run / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    metadata = {
                        "run": experiment_run,
                        "scene": scene_name,
                        "direction": scene_direction,
                        "metadata": metadata,
                        "distance": calculate_drone_distance(metadata)
                    }
                    all_metadata.append(metadata)
                except json.JSONDecodeError as e:
                    print(f"Error reading {metadata_file}: {e}")
            else:
                print(f"Missing metadata.json for run {experiment_run} in {scene_name}")

    return pd.DataFrame(all_metadata)


def load_parquet_data(results_dir: str | Path, scene_dir_names: list[str], is_student: bool = False):
    all_logs = []
    results_dir = Path(results_dir)

    parquet_log_name = "logs.parquet"
    for scene_name in scene_dir_names:
        parquet_logs_path = results_dir / scene_name / "parquet-logs"
        if not parquet_logs_path.exists():
            print(f"No results directory for {scene_name} check the path ({parquet_logs_path})")
            continue

        scene_direction = extract_direction(scene_name)
        for experiment_run in os.listdir(parquet_logs_path):
            parquet_file = parquet_logs_path / experiment_run / parquet_log_name
            if not parquet_file.exists():
                print(f"No parquet file for {parquet_file} in {experiment_run}")
                continue

            df = pd.read_parquet(parquet_file)

            if is_student:
                parquet_teacher_output = parquet_logs_path / experiment_run / "logs-teacher-output.parquet"
                if not parquet_teacher_output.exists():
                    print(f"Non existent parquet file: {parquet_teacher_output}")
                    continue
                try:
                    df_teacher = pd.read_parquet(parquet_teacher_output)
                    df_teacher = df_teacher[[
                        'timestamp', 'frame_idx', 'jacobian_matrix', 'jcond',
                        'current_points_flatten', 'goal_points_flatten', 'err_uv', 'velocity'
                    ]]
                except Exception as e:
                    print(f"Error reading parquet file:\n{e}\n{parquet_teacher_output}")
                    continue
                df = df.merge(df_teacher, on=['timestamp', 'frame_idx'], how='inner')

            df["run"] = experiment_run
            df["scene"] = scene_name
            df["direction"] = scene_direction
            all_logs.append(df)

    return pd.concat(all_logs, ignore_index=True)


if __name__ == '__main__':
    BASE_DIR = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/real-world-ibvs-results")
    BASE_DIR = Path("/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged")

    pd.set_option('display.max_columns', None)

    data = load_json_flight_data(BASE_DIR, ConfigsDirName.REAL)
    print(data.head)
    print(data.columns)
    print(data['direction'].value_counts())

    data_parquet = load_parquet_data(BASE_DIR, ConfigsDirName.REAL)
    print(data_parquet.head)
    print(data_parquet.columns)

    metadata_df = load_json_metadata(BASE_DIR, ConfigsDirName.REAL)
    print(metadata_df.head())
    print(metadata_df.columns)
