# import json
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
#
# # sns.set(style="whitegrid")
#
# BASE_DIR = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output"
# CONFIG_DIRS = [
#     "bunker-online-4k-config-test-down-left",
#     "bunker-online-4k-config-test-down-right",
#
#     "bunker-online-4k-config-test-left",
#     "bunker-online-4k-config-test-right",
#
#     "bunker-online-4k-config-test-up-left",
#     "bunker-online-4k-config-test-up-right",
#
#     "bunker-online-4k-config-test-front-small-offset-right",
#     "bunker-online-4k-config-test-front-small-offset-left"
# ]
#
#
# def load_flight_durations():
#     all_data = []
#     for config in CONFIG_DIRS:
#         folder = os.path.join(BASE_DIR, config, "results")
#         if not os.path.exists(folder):
#             continue
#
#         for run in os.listdir(folder):
#             path = os.path.join(folder, run, "flight_duration.json")
#             if os.path.exists(path):
#                 with open(path) as f:
#                     data = json.load(f)
#                     data["run"] = run
#                     data["config"] = config
#
#                     if "down-left" in config:
#                         data["direction"] = "down-left"
#                     elif "down-right" in config:
#                         data["direction"] = "down-right"
#                     elif "up-left" in config:
#                         data["direction"] = "up-left"
#                     elif "up-right" in config:
#                         data["direction"] = "up-right"
#                     elif config.endswith("left"):
#                         data["direction"] = "left"
#                     elif config.endswith("right"):
#                         data["direction"] = "right"
#                     else:
#                         data["direction"] = "unknown"
#
#                     all_data.append(data)
#     return pd.DataFrame(all_data)
#
#
# def load_parquet_logs():
#     all_logs = []
#     for config in CONFIG_DIRS:
#         folder = os.path.join(BASE_DIR, config, "parquet-logs")
#         if not os.path.exists(folder):
#             continue
#
#         for run in os.listdir(folder):
#             parquet_file = os.path.join(folder, run, "logs.parquet")
#             if os.path.exists(parquet_file):
#                 df = pd.read_parquet(parquet_file)
#                 df["run"] = run
#                 df["config"] = config
#
#                 if "down-left" in config:
#                     df["direction"] = "down-left"
#                 elif "down-right" in config:
#                     df["direction"] = "down-right"
#                 elif "up-left" in config:
#                     df["direction"] = "up-left"
#                 elif "up-right" in config:
#                     df["direction"] = "up-right"
#                 elif config.endswith("left"):
#                     df["direction"] = "left"
#                 elif config.endswith("right"):
#                     df["direction"] = "right"
#                 else:
#                     df["direction"] = "unknown"
#
#                 all_logs.append(df)
#     return pd.concat(all_logs, ignore_index=True)
#
#
# def extract_vector_norms(df):
#     """Extract norms from vector columns"""
#     if 'err_uv' in df.columns:
#         df['err_norm'] = df['err_uv'].apply(
#             lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) else np.nan
#         )
#
#     if 'velocity' in df.columns:
#         df['velocity_norm'] = df['velocity'].apply(
#             lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) else np.nan
#         )
#
#     command_cols = ['x_cmd', 'y_cmd', 'z_cmd', 'rot_cmd']
#     available_cmd_cols = [col for col in command_cols if col in df.columns]
#     if available_cmd_cols:
#         df['cmd_norm'] = df[available_cmd_cols].apply(lambda row: np.linalg.norm(row.values), axis=1)
#
#     return df
#
# def main():
#     print("Loading flight durations...")
#     durations_df = load_flight_durations()
#     print(durations_df.head())
#
#     # Plot flight durations
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=durations_df, x="run", y="flight_duration", hue="config")
#     plt.title("Flight Duration per Run")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("flight_durations.png")
#     plt.close()
#
#     print("Loading parquet logs...")
#     logs_df = load_parquet_logs()
#     print(logs_df.head())
#
#     # for cmd in ["x_cmd", "y_cmd", "z_cmd", "rot_cmd"]:
#     #     plt.figure(figsize=(12, 6))
#     #     sns.lineplot(data=logs_df, x="jcond", y=cmd, hue="config")
#     #     plt.title(f"{cmd} Over Jacobian Condition Number")
#     #     plt.tight_layout()
#     #     plt.savefig(f"{cmd}_jacobian.png")
#     #     plt.close()
#
#     # Plot Jacobian condition number
#     plt.figure(figsize=(10, 6))
#     sns.histplot(logs_df["jcond"], bins=50, kde=True)
#     plt.title("Jacobian Condition Number Distribution Over All Runs")
#     plt.tight_layout()
#     plt.savefig("jcond_distribution.png")
#     plt.close()
#
#     # Error vector norm (err_uv)
#     logs_df["err_norm"] = logs_df["err_uv"].apply(lambda x: np.linalg.norm(x))
#     print(logs_df["err_norm"])
#
#     plt.figure(figsize=(12, 6))
#     sns.histplot(logs_df["err_norm"], bins=50, kde=True)
#     plt.title("Tracking Error Norm Over All Runs")
#     plt.tight_layout()
#     plt.savefig("error_norm_runs.png")
#     plt.close()
#
#     sample_run = logs_df[logs_df['run'] == '2025-06-11_13-39-28']
#     fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
#     sns.lineplot(data=sample_run, x="timestamp", y="x_cmd", ax=ax[0])
#     sns.lineplot(data=sample_run, x="timestamp", y="y_cmd", ax=ax[1])
#     sns.lineplot(data=sample_run, x="timestamp", y="rot_cmd", ax=ax[2])
#     sns.lineplot(data=sample_run, x="timestamp", y="err_norm", ax=ax[3])
#     plt.tight_layout()
#     plt.savefig("time_series_sample_run.png")
#     plt.close()
#
#     # cam garbage
#     sns.scatterplot(data=logs_df, x="jcond", y="err_norm", alpha=0.3)
#     plt.title("Jacobian Condition Number vs. Tracking Error")
#     plt.savefig("jcond_vs_error.png")
#     plt.close()
#
#     print("Analysis complete. Plots saved in current directory.")
#
#
# if __name__ == "__main__":
#     main()
