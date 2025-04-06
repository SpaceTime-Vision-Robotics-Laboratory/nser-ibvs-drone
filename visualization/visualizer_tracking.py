#!/usr/bin/env python3
# tracking_visualizer.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

def visualize_tracking_data(csv_file, output_dir=None):
    """
    Create visualizations from drone tracking data.
    Args:
        csv_file: Path to the tracking data CSV file
        output_dir: Directory to save plots (or None to display)
    """
    # Load the tracking data
    print(f"Loading tracking data from {csv_file}")
    df = pd.read_csv(csv_file)

    # Convert timestamp to datetime for better x-axis labels
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to {output_dir}")

    # Set up the plotting style
    plt.style.use('dark_background')

    # 1. Create a comprehensive overview plot
    create_overview_plot(df, output_dir)

    # 2. Create a positional tracking plot
    create_position_plot(df, output_dir)

    # 3. Create a size ratio plot
    create_size_plot(df, output_dir)

    # 4. Create a commands plot
    create_commands_plot(df, output_dir)

    # 5. Create a heatmap of object positions
    create_position_heatmap(df, output_dir)

    # If not saving to files, show the plots
    if not output_dir:
        plt.show()

    print("Visualization complete!")

# --- Helper functions for create_overview_plot ---

def _plot_offsets(ax, df):
    """Plots X and Y offsets."""
    ax.plot(df['datetime'], df['x_offset'], 'r-', label='X Offset')
    ax.plot(df['datetime'], df['y_offset'], 'g-', label='Y Offset')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Normalized Offset')
    ax.set_title('Object Offsets from Center')
    ax.legend()
    ax.grid(alpha=0.3)

def _plot_size_ratio_overview(ax, df):
    """Plots size ratio."""
    ax.plot(df['datetime'], df['size_ratio'], 'b-', label='Current Ratio')
    ax.axhline(y=df['target_ratio'].iloc[0], color='cyan', linestyle='--', label='Target Ratio')
    ax.set_ylabel('Size Ratio')
    ax.set_title('Object Size Ratio vs Target')
    ax.legend()
    ax.grid(alpha=0.3)

def _plot_commands_overview(ax, df):
    """Plots drone commands."""
    if 'x_cmd' in df.columns:
        ax.plot(df['datetime'], df['x_cmd'], 'm-', label='X Cmd (Strafe)', alpha=0.7)
        ax.plot(df['datetime'], df['y_cmd'], 'y-', label='Y Cmd (Forward)', alpha=0.7)
        ax.plot(df['datetime'], df['z_cmd'], 'c-', label='Z Cmd (Altitude)', alpha=0.7)
        ax.plot(df['datetime'], df['rot_cmd'], 'w-', label='Rotation Cmd', alpha=0.7)
    else:
        # For compatibility with older data format
        ax.plot(df['datetime'], df['rotation_cmd'], 'r-', label='Rotation', alpha=0.7)
        ax.plot(df['datetime'], df['altitude_cmd'], 'g-', label='Altitude', alpha=0.7)
        ax.plot(df['datetime'], df['forward_cmd'], 'b-', label='Forward', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Command Value')
    ax.set_title('Drone Commands')
    ax.legend()
    ax.grid(alpha=0.3)

def _plot_position_scatter(ax, df):
    """Plots object position scatter."""
    scatter = ax.scatter(
        df['object_center_x'], df['object_center_y'],
        c=df['timestamp'] - df['timestamp'].iloc[0],
        cmap='viridis', alpha=0.6, s=10
    )
    ax.scatter(
        [df['frame_center_x'].iloc[0]], [df['frame_center_y'].iloc[0]],
        color='red', marker='+', s=100, label='Frame Center'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (seconds)')
    ax.set_xlim(0, df['frame_center_x'].iloc[0] * 2)
    ax.set_ylim(0, df['frame_center_y'].iloc[0] * 2)
    ax.set_title('Object Position')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.legend()
    ax.grid(alpha=0.3)

def _plot_box_dimensions(ax, df):
    """Plots bounding box dimensions."""
    ax.plot(df['datetime'], df['box_width'], 'orange', label='Width')
    ax.plot(df['datetime'], df['box_height'], 'cyan', label='Height')
    ax.set_ylabel('Pixels')
    ax.set_title('Bounding Box Dimensions')
    ax.legend()
    ax.grid(alpha=0.3)

# --- Main plotting functions ---

def create_overview_plot(df, output_dir):
    """Create a comprehensive overview of all tracking metrics using helper functions."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)

    # Create axes
    ax1 = fig.add_subplot(gs[0, :2]) # Offsets
    ax2 = fig.add_subplot(gs[1, :2]) # Size Ratio
    ax3 = fig.add_subplot(gs[2, :2]) # Commands
    ax4 = fig.add_subplot(gs[:2, 2]) # Position Scatter
    ax5 = fig.add_subplot(gs[2, 2]) # Box Dimensions

    # Call helper functions to plot on respective axes
    _plot_offsets(ax1, df)
    _plot_size_ratio_overview(ax2, df)
    _plot_commands_overview(ax3, df)
    _plot_position_scatter(ax4, df)
    _plot_box_dimensions(ax5, df)

    # Format the datetime x-axis for relevant plots
    for ax in [ax1, ax2, ax3, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_xlabel('Time (MM:SS)')

    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/tracking_overview.png", dpi=200)
        plt.close()

def create_position_plot(df, output_dir):
    """Create a plot showing object position over time"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create points with time-based color
    points = ax.scatter(
        df['object_center_x'],
        df['object_center_y'],
        c=df['timestamp'] - df['timestamp'].iloc[0],
        cmap='plasma',
        s=15,
        alpha=0.7
    )

    # Add frame center
    ax.scatter(
        [df['frame_center_x'].iloc[0]],
        [df['frame_center_y'].iloc[0]],
        color='lime',
        marker='+',
        s=200,
        label='Frame Center'
    )

    # Add arrow to show movement direction
    for i in range(0, len(df), max(1, len(df) // 20)):  # Add arrows at intervals
        if i < len(df) - 1:
            ax.arrow(
                df['object_center_x'].iloc[i],
                df['object_center_y'].iloc[i],
                df['object_center_x'].iloc[i+1] - df['object_center_x'].iloc[i],
                df['object_center_y'].iloc[i+1] - df['object_center_y'].iloc[i],
                head_width=10, head_length=10, fc='white', ec='white', alpha=0.5
            )

    # Add colorbar for time
    cbar = plt.colorbar(points)
    cbar.set_label('Time (seconds)')

    # Set axis limits to frame dimensions
    ax.set_xlim(0, df['frame_center_x'].iloc[0] * 2)
    ax.set_ylim(0, df['frame_center_y'].iloc[0] * 2)

    # Add grid
    ax.grid(alpha=0.3)

    # Labels
    ax.set_title('Object Position Trajectory')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.legend()

    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/position_trajectory.png", dpi=200)
        plt.close()

def create_size_plot(df, output_dir):
    """Create a plot showing size ratio and error over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Size ratio plot
    ax1.plot(df['datetime'], df['size_ratio'], 'b-', label='Size Ratio')
    ax1.axhline(y=df['target_ratio'].iloc[0], color='cyan', linestyle='--', label=f'Target ({df["target_ratio"].iloc[0]:.2f})')

    # Add tolerance band
    # Calculate tolerance safely, handling potential NaN or zero target_ratio
    target_ratio_val = df['target_ratio'].iloc[0] if pd.notna(df['target_ratio'].iloc[0]) else 0
    tolerance = 0.15 * target_ratio_val if target_ratio_val != 0 else 0.01 # Avoid zero tolerance

    ax1.fill_between(
        df['datetime'],
        target_ratio_val - tolerance,
        target_ratio_val + tolerance,
        color='cyan',
        alpha=0.1,
        label='±15% Tolerance'
    )

    ax1.set_ylabel('Size Ratio')
    ax1.set_title('Object Size Ratio vs Target')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Size error plot
    ax2.plot(df['datetime'], df['size_error'], 'r-', label='Size Error')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(
        df['datetime'],
        -0.2, 0.2,
        color='green',
        alpha=0.1,
        label='Acceptable Range'
    )

    ax2.set_ylabel('Normalized Error')
    ax2.set_title('Size Error (positive = too far, negative = too close)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Format the datetime x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    ax2.set_xlabel('Time (MM:SS)')

    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/size_analysis.png", dpi=200)
        plt.close()

def create_commands_plot(df, output_dir):
    """Create a plot showing commands over time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine which command columns to use
    if 'x_cmd' in df.columns:
        ax.plot(df['datetime'], df['x_cmd'], 'm-', label='X (Strafe)', alpha=0.7)
        ax.plot(df['datetime'], df['y_cmd'], 'y-', label='Y (Forward)', alpha=0.7)
        ax.plot(df['datetime'], df['z_cmd'], 'c-', label='Z (Altitude)', alpha=0.7)
        ax.plot(df['datetime'], df['rot_cmd'], 'w-', label='Rotation', alpha=0.7)
    else:
        # For compatibility with older data format
        ax.plot(df['datetime'], df['rotation_cmd'], 'r-', label='Rotation', alpha=0.7)
        ax.plot(df['datetime'], df['altitude_cmd'], 'g-', label='Altitude', alpha=0.7)
        ax.plot(df['datetime'], df['forward_cmd'], 'b-', label='Forward', alpha=0.7)

    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add command limits
    ax.axhline(y=100, color='red', linestyle='-.', alpha=0.3)
    ax.axhline(y=-100, color='red', linestyle='-.', alpha=0.3)

    # Format the datetime x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    ax.set_xlabel('Time (MM:SS)')

    ax.set_ylabel('Command Value')
    ax.set_title('Drone Control Commands')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/commands.png", dpi=200)
        plt.close()

def create_position_heatmap(df, output_dir):
    """Create a heatmap showing where the object spent most time"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a 2D histogram (heatmap)
    # Handle potential missing frame center data
    frame_center_x = df['frame_center_x'].iloc[0] if 'frame_center_x' in df.columns and pd.notna(df['frame_center_x'].iloc[0]) else 640 # Default fallback
    frame_center_y = df['frame_center_y'].iloc[0] if 'frame_center_y' in df.columns and pd.notna(df['frame_center_y'].iloc[0]) else 360 # Default fallback
    frame_width = frame_center_x * 2
    frame_height = frame_center_y * 2

    bins = [max(1, int(frame_width / 20)), max(1, int(frame_height / 20))] # Ensure bins >= 1

    # Filter out NaN positions before histogramming
    valid_pos_df = df.dropna(subset=['object_center_x', 'object_center_y'])

    if valid_pos_df.empty:
        print("Warning: No valid object positions found for heatmap.")
        ax.text(0.5, 0.5, "No Position Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Object Position Heatmap (No Data)')
    else:
        heatmap, xedges, yedges = np.histogram2d(
            valid_pos_df['object_center_x'],
            valid_pos_df['object_center_y'],
            bins=bins,
            range=[[0, frame_width], [0, frame_height]]
        )

        # Smoothing for better visualization
        heatmap = gaussian_filter(heatmap, sigma=1.5)

        # Create the heatmap
        img = ax.imshow(
            heatmap.T,  # Transpose for correct orientation
            origin='lower',
            extent=[0, frame_width, 0, frame_height],
            cmap='hot',
            aspect='auto'
        )

        # Add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Time Density')
        ax.set_title('Object Position Heatmap')


    # Plot the frame center
    ax.scatter(
        [frame_center_x],
        [frame_center_y],
        color='cyan',
        marker='+',
        s=200,
        label='Frame Center'
    )

    # Add labels
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.legend()

    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/position_heatmap.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize drone tracking data')
    parser.add_argument('csv_file', help='Path to the tracking data CSV file')
    parser.add_argument('--output', '-o', help='Directory to save plots (optional)')

    args = parser.parse_args()

    visualize_tracking_data(args.csv_file, args.output)
