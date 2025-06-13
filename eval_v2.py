import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
    """Load flight duration data from all configurations"""
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
                    # Extract direction from config name
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
    """Load and combine all parquet log files"""
    all_logs = []
    for config in CONFIG_DIRS:
        folder = os.path.join(BASE_DIR, config, "parquet-logs")
        if not os.path.exists(folder):
            continue
        for run in os.listdir(folder):
            parquet_file = os.path.join(folder, run, "logs.parquet")
            if os.path.exists(parquet_file):
                try:
                    df = pd.read_parquet(parquet_file)
                    df["run"] = run
                    df["config"] = config
                    # Extract direction
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
                except Exception as e:
                    print(f"Error loading {parquet_file}: {e}")

    if all_logs:
        return pd.concat(all_logs, ignore_index=True)
    else:
        return pd.DataFrame()


def extract_vector_norms(df):
    """Extract norms from vector columns"""
    # Error norm
    if 'err_uv' in df.columns:
        df['err_norm'] = df['err_uv'].apply(
            lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) else np.nan)

    # Velocity norm
    if 'velocity' in df.columns:
        df['velocity_norm'] = df['velocity'].apply(
            lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) else np.nan)

    # Command norms
    command_cols = ['x_cmd', 'y_cmd', 'z_cmd']
    available_cmd_cols = [col for col in command_cols if col in df.columns]
    if available_cmd_cols:
        df['cmd_norm'] = df[available_cmd_cols].apply(lambda row: np.linalg.norm(row.values), axis=1)

    return df


def analyze_flight_durations(durations_df):
    """Comprehensive flight duration analysis"""
    print("=== FLIGHT DURATION ANALYSIS ===")

    if durations_df.empty:
        print("No flight duration data available")
        return

    # Basic statistics
    print(f"Total flights: {len(durations_df)}")
    print(f"Successful flights: {sum(durations_df['status'].isin(['complete', 'complete-soft']))}")
    print(f"Success rate: {sum(durations_df['status'].isin(['complete', 'complete-soft'])) / len(durations_df) * 100:.1f}%")
    print(f"Average flight duration: {durations_df['flight_duration'].mean():.2f}s")
    print(f"Duration std: {durations_df['flight_duration'].std():.2f}s")

    # Success rate by direction
    success_by_direction = durations_df.groupby('direction').agg({
        'status': lambda x: sum(x.isin(['complete', 'complete-soft'])) / len(x) * 100,
        'flight_duration': ['mean', 'std', 'count']
    }).round(2)
    print("\nSuccess rate and duration by direction:")
    print(success_by_direction)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Flight duration by direction
    sns.boxplot(data=durations_df, x='direction', y='flight_duration', ax=axes[0, 0])
    axes[0, 0].set_title('Flight Duration Distribution by Direction')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Success rate by direction
    success_rates = durations_df.groupby('direction')['status'].apply(lambda x: sum(x.isin(['complete', 'complete-soft'])) / len(x) * 100)
    sns.barplot(x=success_rates.index, y=success_rates.values, ax=axes[0, 1])
    axes[0, 1].set_title('Success Rate by Direction (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Duration histogram
    sns.histplot(durations_df['flight_duration'], bins=20, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Flight Duration Distribution')

    # Timeline of flights
    durations_df['datetime'] = pd.to_datetime(durations_df['run'], format='%Y-%m-%d_%H-%M-%S')
    sns.scatterplot(data=durations_df, x='datetime', y='flight_duration', hue='direction', ax=axes[1, 1])
    axes[1, 1].set_title('Flight Duration Over Time')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('flight_duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_temporal_patterns(logs_df):
    """Analyze how performance changes over time during flights"""
    print("\n=== TEMPORAL ANALYSIS ===")

    if logs_df.empty:
        print("No log data available")
        return

    # Normalize timestamp within each flight
    logs_df['flight_progress'] = logs_df.groupby(['config', 'run'])['timestamp'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )

    # Create temporal analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Error evolution during flight
    if 'err_norm' in logs_df.columns:
        # Bin flight progress for better visualization
        logs_df['progress_bin'] = pd.cut(logs_df['flight_progress'], bins=10, labels=False)
        error_evolution = logs_df.groupby('progress_bin')['err_norm'].agg(['mean', 'std']).reset_index()
        error_evolution['progress_bin'] = error_evolution['progress_bin'] * 0.1 + 0.05  # Convert to percentage

        axes[0, 0].errorbar(error_evolution['progress_bin'], error_evolution['mean'],
                            yerr=error_evolution['std'], marker='o', capsize=5)
        axes[0, 0].set_title('Tracking Error Evolution During Flight')
        axes[0, 0].set_xlabel('Flight Progress (0=start, 1=end)')
        axes[0, 0].set_ylabel('Error Norm')

    # Jacobian condition number evolution
    if 'jcond' in logs_df.columns:
        jcond_evolution = logs_df.groupby('progress_bin')['jcond'].agg(['mean', 'std']).reset_index()
        jcond_evolution['progress_bin'] = jcond_evolution['progress_bin'] * 0.1 + 0.05

        axes[0, 1].errorbar(jcond_evolution['progress_bin'], jcond_evolution['mean'],
                            yerr=jcond_evolution['std'], marker='s', capsize=5, color='orange')
        axes[0, 1].set_title('Jacobian Condition Number Evolution')
        axes[0, 1].set_xlabel('Flight Progress')
        axes[0, 1].set_ylabel('Condition Number')

    # Velocity evolution
    if 'velocity_norm' in logs_df.columns:
        vel_evolution = logs_df.groupby('progress_bin')['velocity_norm'].agg(['mean', 'std']).reset_index()
        vel_evolution['progress_bin'] = vel_evolution['progress_bin'] * 0.1 + 0.05

        axes[0, 2].errorbar(vel_evolution['progress_bin'], vel_evolution['mean'],
                            yerr=vel_evolution['std'], marker='^', capsize=5, color='green')
        axes[0, 2].set_title('Velocity Evolution During Flight')
        axes[0, 2].set_xlabel('Flight Progress')
        axes[0, 2].set_ylabel('Velocity Norm')

    # Command magnitude evolution
    if 'cmd_norm' in logs_df.columns:
        cmd_evolution = logs_df.groupby('progress_bin')['cmd_norm'].agg(['mean', 'std']).reset_index()
        cmd_evolution['progress_bin'] = cmd_evolution['progress_bin'] * 0.1 + 0.05

        axes[1, 0].errorbar(cmd_evolution['progress_bin'], cmd_evolution['mean'],
                            yerr=cmd_evolution['std'], marker='d', capsize=5, color='red')
        axes[1, 0].set_title('Control Command Evolution')
        axes[1, 0].set_xlabel('Flight Progress')
        axes[1, 0].set_ylabel('Command Norm')

    # Time series example (first flight)
    if len(logs_df) > 0:
        sample_flight = logs_df[logs_df['run'] == logs_df['run'].iloc[0]]
        if 'err_norm' in sample_flight.columns:
            axes[1, 1].plot(sample_flight['flight_progress'], sample_flight['err_norm'], alpha=0.7)
            axes[1, 1].set_title('Sample Flight: Error vs Time')
            axes[1, 1].set_xlabel('Flight Progress')
            axes[1, 1].set_ylabel('Error Norm')

    # Control effort distribution by flight phase
    if 'cmd_norm' in logs_df.columns:
        phase_labels = ['Start', 'Early', 'Mid-Early', 'Mid', 'Mid-Late', 'Late', 'End']
        logs_df['flight_phase'] = pd.cut(logs_df['flight_progress'], bins=7, labels=phase_labels)
        sns.boxplot(data=logs_df, x='flight_phase', y='cmd_norm', ax=axes[1, 2])
        axes[1, 2].set_title('Control Effort by Flight Phase')
        axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_configuration_performance(logs_df, durations_df):
    """Compare performance across different configurations"""
    print("\n=== CONFIGURATION COMPARISON ===")

    if logs_df.empty:
        print("No log data available")
        return

    # Performance metrics by direction
    if 'err_norm' in logs_df.columns:
        perf_metrics = logs_df.groupby('direction').agg({
            'err_norm': ['mean', 'std', 'median'],
            'jcond': ['mean', 'std', 'median'] if 'jcond' in logs_df.columns else lambda x: np.nan,
            'velocity_norm': ['mean', 'std'] if 'velocity_norm' in logs_df.columns else lambda x: np.nan,
            'cmd_norm': ['mean', 'std'] if 'cmd_norm' in logs_df.columns else lambda x: np.nan
        }).round(3)

        print("Performance metrics by direction:")
        print(perf_metrics)

    # Create comparison visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Error comparison
    if 'err_norm' in logs_df.columns:
        sns.boxplot(data=logs_df, x='direction', y='err_norm', ax=axes[0, 0])
        axes[0, 0].set_title('Tracking Error by Direction')
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Jacobian condition number comparison
    if 'jcond' in logs_df.columns:
        sns.boxplot(data=logs_df, x='direction', y='jcond', ax=axes[0, 1])
        axes[0, 1].set_title('Jacobian Condition Number by Direction')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_yscale('log')  # Log scale for condition number

    # Velocity comparison
    if 'velocity_norm' in logs_df.columns:
        sns.boxplot(data=logs_df, x='direction', y='velocity_norm', ax=axes[0, 2])
        axes[0, 2].set_title('Velocity by Direction')
        axes[0, 2].tick_params(axis='x', rotation=45)

    # Control effort comparison
    if 'cmd_norm' in logs_df.columns:
        sns.boxplot(data=logs_df, x='direction', y='cmd_norm', ax=axes[1, 0])
        axes[1, 0].set_title('Control Effort by Direction')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Performance consistency (coefficient of variation)
    if 'err_norm' in logs_df.columns:
        consistency = logs_df.groupby('direction')['err_norm'].agg(lambda x: x.std() / x.mean()).sort_values()
        sns.barplot(x=consistency.index, y=consistency.values, ax=axes[1, 1])
        axes[1, 1].set_title('Error Consistency by Direction\n(Lower = More Consistent)')
        axes[1, 1].tick_params(axis='x', rotation=45)

    # Correlation heatmap
    numeric_cols = logs_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = logs_df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Variable Correlation Matrix')

    plt.tight_layout()
    plt.savefig('configuration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_control_system(logs_df):
    """Analyze control system characteristics"""
    print("\n=== CONTROL SYSTEM ANALYSIS ===")

    if logs_df.empty or 'jcond' not in logs_df.columns:
        print("No jacobian data available")
        return

    # Identify ill-conditioned moments
    high_jcond_threshold = logs_df['jcond'].quantile(0.95)
    logs_df['ill_conditioned'] = logs_df['jcond'] > high_jcond_threshold

    print(f"Ill-conditioned moments (jcond > {high_jcond_threshold:.2f}): {logs_df['ill_conditioned'].sum()}")
    print(f"Percentage of ill-conditioned moments: {logs_df['ill_conditioned'].mean() * 100:.2f}%")

    # Analyze relationship between condition number and error
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Jacobian condition vs error
    if 'err_norm' in logs_df.columns:
        # Sample data for scatter plot if too many points
        sample_size = min(10000, len(logs_df))
        sample_df = logs_df.sample(n=sample_size) if len(logs_df) > sample_size else logs_df

        axes[0, 0].scatter(sample_df['jcond'], sample_df['err_norm'], alpha=0.5, s=1)
        axes[0, 0].set_xlabel('Jacobian Condition Number')
        axes[0, 0].set_ylabel('Tracking Error')
        axes[0, 0].set_title('Condition Number vs Tracking Error')
        axes[0, 0].set_xscale('log')

    # Distribution of condition numbers
    sns.histplot(logs_df['jcond'], bins=50, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(high_jcond_threshold, color='red', linestyle='--', label=f'95th percentile')
    axes[0, 1].set_title('Jacobian Condition Number Distribution')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()

    # Error during ill-conditioned vs well-conditioned moments
    if 'err_norm' in logs_df.columns:
        error_comparison = logs_df.groupby('ill_conditioned')['err_norm'].agg(['mean', 'std', 'median'])
        print("\nError comparison (ill-conditioned vs well-conditioned):")
        print(error_comparison)

        sns.boxplot(data=logs_df, x='ill_conditioned', y='err_norm', ax=axes[1, 0])
        axes[1, 0].set_title('Error: Well-conditioned vs Ill-conditioned')
        axes[1, 0].set_xticklabels(['Well-conditioned', 'Ill-conditioned'])

    # Condition number by direction
    sns.boxplot(data=logs_df, x='direction', y='jcond', ax=axes[1, 1])
    axes[1, 1].set_title('Condition Number by Direction')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('control_system_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def statistical_analysis(logs_df, durations_df):
    """Perform statistical tests and advanced analysis"""
    print("\n=== STATISTICAL ANALYSIS ===")

    if logs_df.empty:
        print("No log data available")
        return

    # ANOVA test for error differences between directions
    if 'err_norm' in logs_df.columns and 'direction' in logs_df.columns:
        directions = logs_df['direction'].unique()
        if len(directions) > 1:
            direction_groups = [logs_df[logs_df['direction'] == d]['err_norm'].dropna() for d in directions]
            direction_groups = [g for g in direction_groups if len(g) > 0]  # Remove empty groups

            if len(direction_groups) > 1:
                f_stat, p_value = stats.f_oneway(*direction_groups)
                print(f"ANOVA test for error differences between directions:")
                print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

                if p_value < 0.05:
                    print("Significant differences found between directions")
                else:
                    print("No significant differences found between directions")

    # Correlation analysis
    numeric_cols = ['err_norm', 'jcond', 'velocity_norm', 'cmd_norm']
    available_cols = [col for col in numeric_cols if col in logs_df.columns]

    if len(available_cols) > 1:
        print(f"\nCorrelation analysis for: {available_cols}")
        correlation_matrix = logs_df[available_cols].corr()
        print(correlation_matrix.round(3))

        # Find strongest correlations
        correlations = []
        for i in range(len(available_cols)):
            for j in range(i + 1, len(available_cols)):
                corr_val = correlation_matrix.iloc[i, j]
                correlations.append((available_cols[i], available_cols[j], corr_val))

        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        print("\nStrongest correlations:")
        for var1, var2, corr in correlations[:5]:
            print(f"{var1} <-> {var2}: {corr:.3f}")

    # Performance rankings
    if 'err_norm' in logs_df.columns:
        direction_performance = logs_df.groupby('direction')['err_norm'].agg(['mean', 'median', 'std']).round(4)
        direction_performance['rank_mean'] = direction_performance['mean'].rank()
        direction_performance['rank_median'] = direction_performance['median'].rank()
        #TODO: this but for last 90 frames (3 seconds)
        print("\nDirection performance ranking (by mean error):")
        print(direction_performance.sort_values('rank_mean'))


def create_summary_report(logs_df, durations_df):
    """Create a comprehensive summary report"""
    print("\n" + "=" * 50)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 50)

    # Overall statistics
    if not durations_df.empty:
        total_flights = len(durations_df)
        successful_flights = sum(durations_df['status'].isin(['complete', 'complete-soft']))
        avg_duration = durations_df['flight_duration'].mean()

        print(f"Total Flights Analyzed: {total_flights}")
        print(f"Successful Flights: {successful_flights} ({successful_flights / total_flights * 100:.1f}%)")
        print(f"Average Flight Duration: {avg_duration:.2f}s")

    if not logs_df.empty:
        total_data_points = len(logs_df)
        print(f"Total Data Points: {total_data_points:,}")

        if 'err_norm' in logs_df.columns:
            overall_error = logs_df['err_norm'].mean()
            print(f"Overall Average Tracking Error: {overall_error:.4f}")

        if 'jcond' in logs_df.columns:
            avg_condition = logs_df['jcond'].mean()
            print(f"Average Jacobian Condition Number: {avg_condition:.2f}")

    # Best and worst performing directions
    if not logs_df.empty and 'err_norm' in logs_df.columns:
        direction_performance = logs_df.groupby('direction')['err_norm'].mean().sort_values()
        print(
            f"\nBest Performing Direction: {direction_performance.index[0]} (Error: {direction_performance.iloc[0]:.4f})")
        print(
            f"Worst Performing Direction: {direction_performance.index[-1]} (Error: {direction_performance.iloc[-1]:.4f})")

    print("\nAnalysis complete! Check the generated PNG files for detailed visualizations:")
    print("- flight_duration_analysis.png")
    print("- temporal_analysis.png")
    print("- configuration_comparison.png")
    print("- control_system_analysis.png")


def main():
    """Main analysis function"""
    print("Starting comprehensive drone flight data analysis...")

    # Load data
    print("Loading flight duration data...")
    durations_df = load_flight_durations()

    print("Loading parquet log data...")
    logs_df = load_parquet_logs()

    if not logs_df.empty:
        # Extract vector norms and derived metrics
        logs_df = extract_vector_norms(logs_df)

    # Run all analyses
    analyze_flight_durations(durations_df)
    analyze_temporal_patterns(logs_df)
    analyze_configuration_performance(logs_df, durations_df)
    analyze_control_system(logs_df)
    statistical_analysis(logs_df, durations_df)
    create_summary_report(logs_df, durations_df)


if __name__ == "__main__":
    main()
