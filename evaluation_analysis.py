import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sns.set(style="whitegrid")

BASE_DIR = "/home/sebnae/shared_drive/ws/drone_ws/auto-follow/output"
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
            lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) else np.nan
        )

    if 'velocity' in df.columns:
        df['velocity_norm'] = df['velocity'].apply(
            lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) else np.nan
        )

    command_cols = ['x_cmd', 'y_cmd', 'z_cmd', 'rot_cmd']
    available_cmd_cols = [col for col in command_cols if col in df.columns]
    if available_cmd_cols:
        df['cmd_norm'] = df[available_cmd_cols].apply(lambda row: np.linalg.norm(row.values), axis=1)

    return df

def main():
    print("Loading flight durations...")
    durations_df = load_flight_durations()
    print(durations_df.head())

    # Plot flight durations
    plt.figure(figsize=(10, 6))
    sns.barplot(data=durations_df, x="run", y="flight_duration", hue="config")
    plt.title("Flight Duration per Run")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("flight_durations.png")
    plt.close()

    print("Loading parquet logs...")
    logs_df = load_parquet_logs()
    print(logs_df.head())

    # for cmd in ["x_cmd", "y_cmd", "z_cmd", "rot_cmd"]:
    #     plt.figure(figsize=(12, 6))
    #     sns.lineplot(data=logs_df, x="jcond", y=cmd, hue="config")
    #     plt.title(f"{cmd} Over Jacobian Condition Number")
    #     plt.tight_layout()
    #     plt.savefig(f"{cmd}_jacobian.png")
    #     plt.close()

    # Plot Jacobian condition number
    plt.figure(figsize=(10, 6))
    sns.histplot(logs_df["jcond"], bins=50, kde=True)
    plt.title("Jacobian Condition Number Distribution Over All Runs")
    plt.tight_layout()
    plt.savefig("jcond_distribution.png")
    plt.close()

    # Error vector norm (err_uv) - normalized between 0 and 1
    logs_df["err_norm"] = logs_df["err_uv"].apply(lambda x: np.linalg.norm(x))
    
    # Normalize error between 0 and 1
    err_min = logs_df["err_norm"].min()
    err_max = logs_df["err_norm"].max()
    if err_max > err_min:
        logs_df["err_norm"] = (logs_df["err_norm"] - err_min) / (err_max - err_min)
    else:
        logs_df["err_norm"] = 0.5  # If all values are the same
    
    print(f"Error normalized to range: {logs_df['err_norm'].min():.4f} - {logs_df['err_norm'].max():.4f}")
    print(logs_df["err_norm"])

    plt.figure(figsize=(12, 6))
    sns.histplot(logs_df["err_norm"], bins=50, kde=True)
    plt.title("Tracking Error Norm Over All Runs")
    plt.tight_layout()
    plt.savefig("error_norm_runs.png")
    plt.close()

    sample_run = logs_df[logs_df['run'] == '2025-06-11_13-39-28']
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    sns.lineplot(data=sample_run, x="timestamp", y="x_cmd", ax=ax[0])
    sns.lineplot(data=sample_run, x="timestamp", y="y_cmd", ax=ax[1])
    sns.lineplot(data=sample_run, x="timestamp", y="rot_cmd", ax=ax[2])
    sns.lineplot(data=sample_run, x="timestamp", y="err_norm", ax=ax[3])
    plt.tight_layout()
    plt.savefig("time_series_sample_run.png")
    plt.close()

    # cam garbage
    sns.scatterplot(data=logs_df, x="jcond", y="err_norm", alpha=0.3)
    plt.title("Jacobian Condition Number vs. Tracking Error")
    plt.savefig("jcond_vs_error.png")
    plt.close()

    # Jacobian condition number at the end of runs
    print("Extracting final jcond values for each run...")
    final_jcond_values = []
    for run_id in logs_df['run'].unique():
        run_data = logs_df[logs_df['run'] == run_id]
        if len(run_data) > 0:
            # Get the last 50 jcond values for this run (or all available if less than 50)
            last_jcond_values = run_data['jcond'].iloc[-15:]
            final_jcond = last_jcond_values.median()
            final_jcond_values.append({
                'run': run_id,
                'config': run_data['config'].iloc[0],
                'direction': run_data['direction'].iloc[0],
                'final_jcond': final_jcond
            })
    
    final_jcond_df = pd.DataFrame(final_jcond_values)
    
    # Plot histogram of final jcond values
    plt.figure(figsize=(12, 8))
    
    # Create subplot with histogram and box plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram of final jcond values
    sns.histplot(data=final_jcond_df, x="final_jcond", bins=30, kde=True, ax=ax1)
    ax1.set_title("Distribution of Jacobian Condition Number at End of Runs", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Final Jacobian Condition Number", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_final_jcond = final_jcond_df['final_jcond'].mean()
    median_final_jcond = final_jcond_df['final_jcond'].median()
    ax1.axvline(mean_final_jcond, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_final_jcond:.3f}')
    ax1.axvline(median_final_jcond, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_final_jcond:.3f}')
    ax1.legend()
    
    # Box plot by direction/config
    sns.boxplot(data=final_jcond_df, x="direction", y="final_jcond", ax=ax2)
    ax2.set_title("Final Jacobian Condition Number by Direction", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Direction", fontsize=12)
    ax2.set_ylabel("Final Jacobian Condition Number", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("final_jcond_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\n=== FINAL JCOND STATISTICS ===")
    print(f"Mean: {mean_final_jcond:.6f}")
    print(f"Median: {median_final_jcond:.6f}")
    print(f"Std Dev: {final_jcond_df['final_jcond'].std():.6f}")
    print(f"Min: {final_jcond_df['final_jcond'].min():.6f}")
    print(f"Max: {final_jcond_df['final_jcond'].max():.6f}")
    
    # Mean and Std over time for commands and error by config
    print("Analyzing command and error statistics over time by configuration...")
    
    # Normalize timestamps within each run to percentage (0-100)
    logs_df_normalized = logs_df.copy()
    for run_id in logs_df['run'].unique():
        run_mask = logs_df_normalized['run'] == run_id
        run_timestamps = logs_df_normalized.loc[run_mask, 'timestamp']
        if len(run_timestamps) > 1:
            min_time = run_timestamps.min()
            max_time = run_timestamps.max()
            if max_time > min_time:
                normalized_time = 100 * (run_timestamps - min_time) / (max_time - min_time)
                logs_df_normalized.loc[run_mask, 'normalized_time'] = normalized_time
            else:
                logs_df_normalized.loc[run_mask, 'normalized_time'] = 0
        else:
            logs_df_normalized.loc[run_mask, 'normalized_time'] = 0
    
    # Create time bins for aggregation
    time_bins = np.linspace(0, 100, 21)  # 20 bins from 0% to 100%
    logs_df_normalized['time_bin'] = pd.cut(logs_df_normalized['normalized_time'], 
                                          bins=time_bins, labels=False, include_lowest=True)
    
    # Calculate statistics for each config and time bin
    stats_by_config = []
    for config in CONFIG_DIRS:
        config_data = logs_df_normalized[logs_df_normalized['config'] == config]
        if len(config_data) == 0:
            continue
            
        for bin_idx in range(20):  # 20 bins
            bin_data = config_data[config_data['time_bin'] == bin_idx]
            if len(bin_data) > 0:
                time_center = (time_bins[bin_idx] + time_bins[bin_idx + 1]) / 2
                stats_by_config.append({
                    'config': config,
                    'time_percent': time_center,
                    'x_cmd_mean': bin_data['x_cmd'].mean(),
                    'x_cmd_std': bin_data['x_cmd'].std(),
                    'y_cmd_mean': bin_data['y_cmd'].mean(),
                    'y_cmd_std': bin_data['y_cmd'].std(),
                    'rot_cmd_mean': bin_data['rot_cmd'].mean(),
                    'rot_cmd_std': bin_data['rot_cmd'].std(),
                    'err_norm_mean': bin_data['err_norm'].mean(),
                    'err_norm_std': bin_data['err_norm'].std(),
                    'sample_count': len(bin_data)
                })
    
    stats_df = pd.DataFrame(stats_by_config)
    
    # Create comprehensive time series plots
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle('Command and Error Statistics Over Time by Configuration', fontsize=16, fontweight='bold')
    
    commands = ['x_cmd', 'y_cmd', 'rot_cmd', 'err_norm']
    command_labels = ['X Command', 'Y Command', 'Rotation Command', 'Error Norm']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(CONFIG_DIRS)))
    
    for i, (cmd, label) in enumerate(zip(commands, command_labels)):
        # Mean plot
        ax_mean = axes[i, 0]
        for j, config in enumerate(CONFIG_DIRS):
            config_stats = stats_df[stats_df['config'] == config]
            if len(config_stats) > 0:
                config_label = config.replace('bunker-online-4k-config-test-', '')
                ax_mean.plot(config_stats['time_percent'], config_stats[f'{cmd}_mean'], 
                           color=colors[j], marker='o', markersize=3, linewidth=2,
                           label=config_label, alpha=0.8)
        
        ax_mean.set_title(f'{label} - Mean Over Time', fontsize=12, fontweight='bold')
        ax_mean.set_xlabel('Flight Progress (%)', fontsize=10)
        ax_mean.set_ylabel(f'Mean {label}', fontsize=10)
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Std plot
        ax_std = axes[i, 1]
        for j, config in enumerate(CONFIG_DIRS):
            config_stats = stats_df[stats_df['config'] == config]
            if len(config_stats) > 0:
                config_label = config.replace('bunker-online-4k-config-test-', '')
                ax_std.plot(config_stats['time_percent'], config_stats[f'{cmd}_std'], 
                          color=colors[j], marker='s', markersize=3, linewidth=2,
                          label=config_label, alpha=0.8)
        
        ax_std.set_title(f'{label} - Standard Deviation Over Time', fontsize=12, fontweight='bold')
        ax_std.set_xlabel('Flight Progress (%)', fontsize=10)
        ax_std.set_ylabel(f'Std Dev {label}', fontsize=10)
        ax_std.grid(True, alpha=0.3)
        ax_std.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("commands_error_stats_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot focusing on error convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error mean convergence
    for j, config in enumerate(CONFIG_DIRS):
        config_stats = stats_df[stats_df['config'] == config]
        if len(config_stats) > 0:
            config_label = config.replace('bunker-online-4k-config-test-', '')
            ax1.plot(config_stats['time_percent'], config_stats['err_norm_mean'], 
                    color=colors[j], marker='o', markersize=4, linewidth=2.5,
                    label=config_label, alpha=0.8)
    
    ax1.set_title('Error Convergence - Mean Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Flight Progress (%)', fontsize=12)
    ax1.set_ylabel('Mean Error Norm', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Error variability over time
    for j, config in enumerate(CONFIG_DIRS):
        config_stats = stats_df[stats_df['config'] == config]
        if len(config_stats) > 0:
            config_label = config.replace('bunker-online-4k-config-test-', '')
            ax2.plot(config_stats['time_percent'], config_stats['err_norm_std'], 
                    color=colors[j], marker='s', markersize=4, linewidth=2.5,
                    label=config_label, alpha=0.8)
    
    ax2.set_title('Error Variability Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Flight Progress (%)', fontsize=12)
    ax2.set_ylabel('Error Norm Std Dev', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("error_convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create RL-style smooth plots with confidence intervals
    print("Creating smooth command evolution plots with confidence intervals...")
    
    # Calculate more granular statistics for smoother plots
    fine_time_bins = np.linspace(0, 100, 51)  # 50 bins for smoother curves
    logs_df_normalized['fine_time_bin'] = pd.cut(logs_df_normalized['normalized_time'], 
                                                bins=fine_time_bins, labels=False, include_lowest=True)
    
    # Calculate statistics for each config and fine time bin
    fine_stats_by_config = []
    for config in CONFIG_DIRS:
        config_data = logs_df_normalized[logs_df_normalized['config'] == config]
        if len(config_data) == 0:
            continue
            
        for bin_idx in range(50):  # 50 bins
            bin_data = config_data[config_data['fine_time_bin'] == bin_idx]
            if len(bin_data) > 0:
                time_center = (fine_time_bins[bin_idx] + fine_time_bins[bin_idx + 1]) / 2
                
                # Calculate percentiles for confidence intervals
                x_cmd_values = bin_data['x_cmd'].dropna()
                y_cmd_values = bin_data['y_cmd'].dropna()
                rot_cmd_values = bin_data['rot_cmd'].dropna()
                err_norm_values = bin_data['err_norm'].dropna()
                
                fine_stats_by_config.append({
                    'config': config,
                    'time_percent': time_center,
                    'x_cmd_mean': x_cmd_values.mean() if len(x_cmd_values) > 0 else np.nan,
                    'x_cmd_std': x_cmd_values.std() if len(x_cmd_values) > 0 else np.nan,
                    'x_cmd_25': x_cmd_values.quantile(0.25) if len(x_cmd_values) > 0 else np.nan,
                    'x_cmd_75': x_cmd_values.quantile(0.75) if len(x_cmd_values) > 0 else np.nan,
                    'y_cmd_mean': y_cmd_values.mean() if len(y_cmd_values) > 0 else np.nan,
                    'y_cmd_std': y_cmd_values.std() if len(y_cmd_values) > 0 else np.nan,
                    'y_cmd_25': y_cmd_values.quantile(0.25) if len(y_cmd_values) > 0 else np.nan,
                    'y_cmd_75': y_cmd_values.quantile(0.75) if len(y_cmd_values) > 0 else np.nan,
                    'rot_cmd_mean': rot_cmd_values.mean() if len(rot_cmd_values) > 0 else np.nan,
                    'rot_cmd_std': rot_cmd_values.std() if len(rot_cmd_values) > 0 else np.nan,
                    'rot_cmd_25': rot_cmd_values.quantile(0.25) if len(rot_cmd_values) > 0 else np.nan,
                    'rot_cmd_75': rot_cmd_values.quantile(0.75) if len(rot_cmd_values) > 0 else np.nan,
                    'err_norm_mean': err_norm_values.mean() if len(err_norm_values) > 0 else np.nan,
                    'err_norm_std': err_norm_values.std() if len(err_norm_values) > 0 else np.nan,
                    'err_norm_25': err_norm_values.quantile(0.25) if len(err_norm_values) > 0 else np.nan,
                    'err_norm_75': err_norm_values.quantile(0.75) if len(err_norm_values) > 0 else np.nan,
                    'sample_count': len(bin_data)
                })
    
    fine_stats_df = pd.DataFrame(fine_stats_by_config)
    
    # Create RL-style plots for each command
    commands_to_plot = [
        ('x_cmd', 'X Command Evolution', 'X Command Value'),
        ('y_cmd', 'Y Command Evolution', 'Y Command Value'), 
        ('rot_cmd', 'Rotation Command Evolution', 'Rotation Command Value'),
        ('err_norm', 'Error Evolution', 'Error Norm')
    ]
    
    # Define a nice color palette
    config_colors = {
        'bunker-online-4k-config-test-down-left': '#1f77b4',
        'bunker-online-4k-config-test-down-right': '#ff7f0e', 
        'bunker-online-4k-config-test-left': '#2ca02c',
        'bunker-online-4k-config-test-right': '#d62728',
        'bunker-online-4k-config-test-up-left': '#9467bd',
        'bunker-online-4k-config-test-up-right': '#8c564b',
        'bunker-online-4k-config-test-front-small-offset-right': '#e377c2',
        'bunker-online-4k-config-test-front-small-offset-left': '#7f7f7f'
    }
    
    for cmd, title, ylabel in commands_to_plot:
        plt.figure(figsize=(12, 8))
        
        for config in CONFIG_DIRS:
            config_stats = fine_stats_df[fine_stats_df['config'] == config].dropna(subset=[f'{cmd}_mean'])
            if len(config_stats) == 0:
                continue
                
            config_label = config.replace('bunker-online-4k-config-test-', '')
            color = config_colors.get(config, '#000000')
            
            # Get data
            x = config_stats['time_percent'].values
            y_mean = config_stats[f'{cmd}_mean'].values
            y_std = config_stats[f'{cmd}_std'].values
            y_25 = config_stats[f'{cmd}_25'].values
            y_75 = config_stats[f'{cmd}_75'].values
            
            # Plot mean line
            plt.plot(x, y_mean, color=color, linewidth=2.5, label=config_label, alpha=0.9)
            
            # Plot confidence interval using standard deviation
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, 
                           color=color, alpha=0.2)
            
            # Alternative: use percentiles for confidence interval (uncomment if preferred)
            # plt.fill_between(x, y_25, y_75, color=color, alpha=0.2)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Flight Progress (%)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.legend(fontsize=12, loc='best')
        
        # Set axis limits
        if cmd in ['x_cmd', 'y_cmd', 'rot_cmd']:
            plt.ylim(-25, 25)
        elif cmd == 'err_norm':
            plt.ylim(0, 1)
        
        # Style similar to RL plot
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1)
        plt.gca().spines['bottom'].set_linewidth(1)
        
        plt.tight_layout()
        plt.savefig(f"{cmd}_evolution_rl_style.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined plot similar to the RL style showing all commands
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (cmd, title, ylabel) in enumerate(commands_to_plot):
        ax = axes[i]
        
        for config in CONFIG_DIRS:
            config_stats = fine_stats_df[fine_stats_df['config'] == config].dropna(subset=[f'{cmd}_mean'])
            if len(config_stats) == 0:
                continue
                
            config_label = config.replace('bunker-online-4k-config-test-', '')
            color = config_colors.get(config, '#000000')
            
            # Get data
            x = config_stats['time_percent'].values
            y_mean = config_stats[f'{cmd}_mean'].values
            y_std = config_stats[f'{cmd}_std'].values
            
            # Plot mean line
            ax.plot(x, y_mean, color=color, linewidth=2, label=config_label, alpha=0.9)
            
            # Plot confidence interval
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                          color=color, alpha=0.2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Flight Progress (%)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set axis limits
        if cmd in ['x_cmd', 'y_cmd', 'rot_cmd']:
            ax.set_ylim(-25, 25)
        elif cmd == 'err_norm':
            ax.set_ylim(0, 1)
        
        # Style similar to RL plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        
        if i == 0:  # Only show legend on first subplot
            ax.legend(fontsize=10, loc='best')
    
    plt.suptitle('Command and Error Evolution (RL Style)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("commands_evolution_rl_style_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print convergence statistics
    print(f"\n=== COMMAND AND ERROR CONVERGENCE ANALYSIS ===")
    for config in CONFIG_DIRS:
        config_stats = stats_df[stats_df['config'] == config]
        if len(config_stats) >= 2:
            config_label = config.replace('bunker-online-4k-config-test-', '')
            
            # Compare first 20% vs last 20% of flight - select only numeric columns
            numeric_cols = ['x_cmd_mean', 'y_cmd_mean', 'rot_cmd_mean', 'err_norm_mean']
            early_stats = config_stats[config_stats['time_percent'] <= 20][numeric_cols].mean()
            late_stats = config_stats[config_stats['time_percent'] >= 80][numeric_cols].mean()
            
            print(f"\n{config_label}:")
            print(f"  Error reduction: {early_stats['err_norm_mean']:.4f} → {late_stats['err_norm_mean']:.4f}")
            print(f"  X cmd reduction: {abs(early_stats['x_cmd_mean']):.4f} → {abs(late_stats['x_cmd_mean']):.4f}")
            print(f"  Y cmd reduction: {abs(early_stats['y_cmd_mean']):.4f} → {abs(late_stats['y_cmd_mean']):.4f}")
            print(f"  Rot cmd reduction: {abs(early_stats['rot_cmd_mean']):.4f} → {abs(late_stats['rot_cmd_mean']):.4f}")

    print("Analysis complete. Plots saved in current directory.")
    print("Generated files:")
    print("- flight_durations.png")
    print("- jcond_distribution.png") 
    print("- error_norm_runs.png")
    print("- time_series_sample_run.png")
    print("- jcond_vs_error.png")
    print("- final_jcond_analysis.png")
    print("- commands_error_stats_over_time.png")
    print("- error_convergence_analysis.png")
    print("- x_cmd_evolution_rl_style.png")
    print("- y_cmd_evolution_rl_style.png")
    print("- rot_cmd_evolution_rl_style.png")
    print("- err_norm_evolution_rl_style.png")
    print("- commands_evolution_rl_style_combined.png")


if __name__ == "__main__":
    main()
