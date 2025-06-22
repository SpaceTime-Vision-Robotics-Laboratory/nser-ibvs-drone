import json
import math

import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm

import numpy as np
from scipy.interpolate import interp1d

from copy import deepcopy

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from auto_follow.utils.path_manager import Paths


def load_drone_data(file_path):
    """Load drone data from JSON file"""
    with open(file_path, 'r') as file:
        content = file.read().strip()
        if not content.endswith(']'):
            content = content.rstrip(',') + ']'
        return json.loads(content)


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between GPS coordinates (Haversine formula)"""
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_drone_distance(log_data):
    """
    Calculate total distance traveled by drone from log data.

    :param log_data: List of dictionaries containing drone telemetry data
    :return: Total distance in meters
    """

    if not log_data or len(log_data) < 2:
        return 0.0

    total_distance = 0.0

    for i in range(1, len(log_data)):
        current_point = log_data[i]
        previous_point = log_data[i - 1]

        speed = current_point['drone']['speed']
        time_delta = current_point['time'] - previous_point['time']
        speed_magnitude = math.sqrt(
            speed['north'] ** 2 +
            speed['east'] ** 2 +
            speed['down'] ** 2
        )

        distance = speed_magnitude * time_delta
        total_distance += distance

    return total_distance


def analyze_trajectory(data, carpet_start_pos: tuple[float, float] | None = None):
    """Analyze drone trajectory and calculate total distance"""
    total_distance = 0.0
    trajectory = []

    has_gps = 'location' in data[0]['drone']

    # If carpet position provided, calculate relative positions
    use_carpet_coords = carpet_start_pos is not None
    if use_carpet_coords:
        carpet_x, carpet_y = carpet_start_pos
        # print(f"Using carpet coordinates with start position: ({carpet_x:.2f}, {carpet_y:.2f}) meters")
    elif not has_gps:
        print("No GPS data found. Use --carpet-start to enable trajectory analysis.")
        return [], 0.0

    for i, point in enumerate(data):
        drone = point['drone']
        speed = drone['speed']

        traj_point = {
            'time': point['time'] - data[0]['time'],  # Relative time
            'speed': math.sqrt(speed['north'] ** 2 + speed['east'] ** 2 + speed['down'] ** 2),
            # 'state': drone['flying_state']
        }

        if has_gps:
            location = drone['location']
            traj_point['lat'] = location['latitude']
            traj_point['lon'] = location['longitude']
            traj_point['alt'] = location['altitude_egm96amsl']
        else:
            # For real drone without GPS, use ground_distance as altitude estimate
            traj_point['alt'] = drone.get('ground_distance', 0.0)

        if use_carpet_coords:
            if i == 0:
                # First point is the starting position on carpet
                traj_point['carpet_x'] = carpet_x
                traj_point['carpet_y'] = carpet_y
            else:
                # Calculate movement using speed and time for carpet coordinates
                prev_point = data[i - 1]
                time_delta = point['time'] - prev_point['time']

                # Use speed components to calculate movement
                dx = speed['east'] * time_delta  # East = +X direction
                dy = speed['north'] * time_delta  # North = +Y direction

                # Add movement to previous carpet position
                prev_carpet_x = trajectory[i - 1]['carpet_x']
                prev_carpet_y = trajectory[i - 1]['carpet_y']

                traj_point['carpet_x'] = prev_carpet_x + dx
                traj_point['carpet_y'] = prev_carpet_y + dy

        if i > 0:
            if use_carpet_coords:
                prev_x = trajectory[i - 1]['carpet_x']
                prev_y = trajectory[i - 1]['carpet_y']
                prev_z = trajectory[i - 1]['alt']

                curr_x = traj_point['carpet_x']
                curr_y = traj_point['carpet_y']
                curr_z = traj_point['alt']

                distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2 + (curr_z - prev_z) ** 2)
            elif has_gps:
                prev_location = data[i - 1]['drone']['location']
                distance = calculate_gps_distance(
                    prev_location['latitude'], prev_location['longitude'],
                    location['latitude'], location['longitude']
                )
            else:
                # No GPS and no carpet coords - use speed integration
                prev_point = data[i - 1]
                time_delta = point['time'] - prev_point['time']
                distance = traj_point['speed'] * time_delta

            total_distance += distance
            traj_point['distance_from_prev'] = distance
        else:
            traj_point['distance_from_prev'] = 0.0

        trajectory.append(traj_point)

    return trajectory, total_distance


def print_results(trajectory, total_distance, use_carpet_coords=False):
    """Print analysis results"""
    if not trajectory:
        print("No trajectory data to analyze.")
        return

    print(f"Total distance: {total_distance:.6f} meters")
    print(f"Flight duration: {trajectory[-1]['time']:.3f} seconds")
    print(f"Data points: {len(trajectory)}")
    print(f"Max speed: {max(p['speed'] for p in trajectory):.3f} m/s")
    print(f"Altitude range: {min(p['alt'] for p in trajectory):.3f}m to {max(p['alt'] for p in trajectory):.3f}m")

    has_gps = 'lat' in trajectory[0] and 'lon' in trajectory[0]

    if use_carpet_coords:
        carpet_xs = [p['carpet_x'] for p in trajectory]
        carpet_ys = [p['carpet_y'] for p in trajectory]
        print(f"Carpet X range: {min(carpet_xs):.3f}m to {max(carpet_xs):.3f}m")
        print(f"Carpet Y range: {min(carpet_ys):.3f}m to {max(carpet_ys):.3f}m")

        # Check if drone stayed within carpet bounds
        out_of_bounds = any(x < 0 or x > 5 or y < 0 or y > 4 for x, y in zip(carpet_xs, carpet_ys))
        if out_of_bounds:
            print("Warning: Drone went outside carpet boundaries (5x4 meters)")

    print("\nTrajectory points:")
    for i, point in enumerate(trajectory):
        if use_carpet_coords:
            print(f"{i + 1:2d}. Time: {point['time']:6.3f}s, "
                  f"Carpet: ({point['carpet_x']:6.3f}, {point['carpet_y']:6.3f}), "
                  f"Alt: {point['alt']:.3f}m, "
                  f"Speed: {point['speed']:.3f}m/s, "
                  f"State: {point['state']}")
        elif has_gps:
            print(f"{i + 1:2d}. Time: {point['time']:6.3f}s, "
                  f"GPS: ({point['lat']:.8f}, {point['lon']:.8f}), "
                  f"Alt: {point['alt']:.3f}m, "
                  f"Speed: {point['speed']:.3f}m/s, "
                  f"State: {point['state']}")
        else:
            print(f"{i + 1:2d}. Time: {point['time']:6.3f}s, "
                  f"Alt: {point['alt']:.3f}m, "
                  f"Speed: {point['speed']:.3f}m/s, "
                  f"State: {point['state']}")


def plot_trajectory(trajectory, goal: tuple[float, float], use_carpet_coords=False):
    """Plot trajectory data"""
    times = [p['time'] for p in trajectory]
    altitudes = [p['alt'] for p in trajectory]
    speeds = [p['speed'] for p in trajectory]

    if use_carpet_coords:
        xs = [p['carpet_x'] for p in trajectory]
        ys = [p['carpet_y'] for p in trajectory]
        x_label, y_label = 'X (m)', 'Y (m)'
        traj_title = 'Carpet Trajectory (5x4m)'
    else:
        xs = [p['lon'] for p in trajectory]
        ys = [p['lat'] for p in trajectory]
        x_label, y_label = 'Longitude', 'Latitude'
        traj_title = 'GPS Trajectory'

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Altitude
    ax1.plot(times, altitudes, 'b-o')
    ax1.set_title('Altitude vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.grid(True)

    # Speed
    ax2.plot(times, speeds, 'r-o')
    ax2.set_title('Speed vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.grid(True)

    # Trajectory (carpet or GPS)
    ax3.plot(xs, ys, 'g-')
    ax3.scatter(xs[0], ys[0], color='blue', s=100, label='Start')
    ax3.scatter(xs[-1], ys[-1], color='red', s=100, label='End')
    print(f"At: {xs[-1], ys[-1]} vs {goal[0] - 0.95, goal[1]}")
    ax3.scatter(goal[0], goal[1], color='green', s=100, label='Car')
    ax3.scatter(goal[0] - 0.95, goal[1], color='magenta', s=100, label='Goal')
    ax3.set_title(traj_title)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.legend()
    ax3.grid(True)

    # Add carpet boundaries if using carpet coordinates
    if use_carpet_coords:
        ax3.add_patch(plt.Rectangle((0, 0), 4, 5, fill=False, edgecolor='black', linewidth=2, linestyle='--'))
        ax3.set_xlim(-0.5, 5.5)
        ax3.set_ylim(-0.5, 6.5)
        ax3.set_aspect('equal')

    # Flight states
    states = [p['state'] for p in trajectory]
    unique_states = list(set(states))
    state_nums = [unique_states.index(state) for state in states]

    ax4.plot(times, state_nums, 's-')
    ax4.set_title('Flight States')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('State')
    ax4.set_yticks(range(len(unique_states)))
    ax4.set_yticklabels([s.replace('FS_', '') for s in unique_states])
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

def plot_trajectory_only(trajectory, goal: tuple[float, float], use_carpet_coords=False):
    """Plot trajectory data"""
    times = [p['time'] for p in trajectory]
    altitudes = [p['alt'] for p in trajectory]
    speeds = [p['speed'] for p in trajectory]

    if use_carpet_coords:
        xs = [p['carpet_x'] for p in trajectory]
        ys = [p['carpet_y'] for p in trajectory]
        x_label, y_label = 'X (m)', 'Y (m)'
        traj_title = 'Carpet Trajectory (5x4m)'
    else:
        xs = [p['lon'] for p in trajectory]
        ys = [p['lat'] for p in trajectory]
        x_label, y_label = 'Longitude', 'Latitude'
        traj_title = 'GPS Trajectory'

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # Trajectory (carpet or GPS)
    ax1.plot(xs, ys, 'g-')
    ax1.scatter(xs[0], ys[0], color='blue', s=100, label='Start')
    ax1.scatter(xs[-1], ys[-1], color='red', s=100, label='End')
    print(f"At: {xs[-1], ys[-1]} vs {goal[0] - 0.95, goal[1]}")
    ax1.scatter(goal[0], goal[1], color='green', s=100, label='Car')
    ax1.scatter(goal[0] - 0.95, goal[1], color='magenta', s=100, label='Goal')
    ax1.set_title(traj_title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.legend()
    ax1.grid(True)

    # Add carpet boundaries if using carpet coordinates
    if use_carpet_coords:
        ax1.add_patch(plt.Rectangle((0, 0), 4, 5, fill=False, edgecolor='black', linewidth=2, linestyle='--'))
        ax1.set_xlim(-0.5, 5.5)
        ax1.set_ylim(-0.5, 6.5)
        ax1.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def main(file_path, goal: tuple[float, float], carpet_start: tuple[float, float] | None = None, plot=True, ):
    data = load_drone_data(file_path)
    print(f"Distance: {calculate_drone_distance(data)}")
    trajectory, total_distance = analyze_trajectory(data, carpet_start)

    use_carpet_coords = carpet_start is not None
    # print_results(trajectory, total_distance, use_carpet_coords)

    if plot:
        # plot_trajectory(trajectory, goal, use_carpet_coords)
        plot_trajectory_only(trajectory, goal, use_carpet_coords)

## -------------------------------------------------------------------------------
## -------------------------------------------------------------------------------

def plot_multiple_trajectories(trajectories, goal: tuple[float, float], use_carpet_coords=False, labels=None):
    """Plot multiple trajectory data on the same plot"""
    
    # Set up labels if not provided
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    # Color cycle for different trajectories
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Determine coordinate system
    if use_carpet_coords:
        x_label, y_label = 'X (m)', 'Y (m)'
        traj_title = 'Multiple Carpet Trajectories (5x4m)'
    else:
        x_label, y_label = 'Longitude', 'Latitude'
        traj_title = 'Multiple GPS Trajectories'
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot each trajectory
    for i, trajectory in enumerate(trajectories):
        if use_carpet_coords:
            xs = [p['carpet_x'] for p in trajectory]
            ys = [p['carpet_y'] for p in trajectory]
        else:
            xs = [p['lon'] for p in trajectory]
            ys = [p['lat'] for p in trajectory]
        
        color = colors[i % len(colors)]
        
        # Plot trajectory line
        ax1.plot(xs, ys, color=color, linewidth=2, label=labels[i])
        
        # Plot start and end points
        ax1.scatter(xs[0], ys[0], color=color, s=100, marker='o', edgecolors='black', linewidth=1)
        ax1.scatter(xs[-1], ys[-1], color=color, s=100, marker='s', edgecolors='black', linewidth=1)
        
        print(f"Trajectory {i+1} end: {xs[-1], ys[-1]} vs goal: {goal[0] - 0.95, goal[1]}")
    
    # Plot goal points
    ax1.scatter(goal[0], goal[1], color='darkgreen', s=150, marker='*', label='Car', edgecolors='black', linewidth=1)
    ax1.scatter(goal[0] - 0.95, goal[1], color='magenta', s=150, marker='*', label='Goal', edgecolors='black', linewidth=1)
    
    # Add legend entries for start/end markers
    ax1.scatter([], [], color='gray', s=100, marker='o', edgecolors='black', linewidth=1, label='Start points')
    ax1.scatter([], [], color='gray', s=100, marker='s', edgecolors='black', linewidth=1, label='End points')
    
    ax1.set_title(traj_title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.legend()
    ax1.grid(True)
    
    # Add carpet boundaries if using carpet coordinates
    if use_carpet_coords:
        ax1.add_patch(plt.Rectangle((0, 0), 4, 5, fill=False, edgecolor='black', linewidth=2, linestyle='--'))
        ax1.set_xlim(-0.5, 5.5)
        ax1.set_ylim(-0.5, 6.5)
        ax1.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def get_mean_std_from_trajectories(trajectories, coord_keys, interpolation_points):
    # Extract and normalize trajectories to same length for averaging
    normalized_trajectories = []
    
    for trajectory in trajectories:
        xs = np.array([p[coord_keys[0]] for p in trajectory])
        ys = np.array([p[coord_keys[1]] for p in trajectory])
        
        if len(xs) < 2:  # Skip trajectories that are too short
            continue
            
        # Create parameter t from 0 to 1
        t_original = np.linspace(0, 1, len(xs))
        t_new = np.linspace(0, 1, interpolation_points)
        
        # Interpolate trajectory to fixed number of points
        try:
            interp_x = interp1d(t_original, xs, kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_y = interp1d(t_original, ys, kind='linear', bounds_error=False, fill_value='extrapolate')
            
            xs_interp = interp_x(t_new)
            ys_interp = interp_y(t_new)
            
            normalized_trajectories.append((xs_interp, ys_interp))
            
        except Exception as e:
            print(f"Skipping trajectory due to interpolation error: {e}")
            continue
    
    if not normalized_trajectories:
        print("No valid trajectories to plot")
        return
    
    # Convert to numpy arrays for easier computation
    all_xs = np.array([traj[0] for traj in normalized_trajectories])
    all_ys = np.array([traj[1] for traj in normalized_trajectories])
    
    # Calculate mean and standard deviation
    mean_xs = np.mean(all_xs, axis=0)
    mean_ys = np.mean(all_ys, axis=0)
    std_xs = np.std(all_xs, axis=0)
    std_ys = np.std(all_ys, axis=0)

    return normalized_trajectories, mean_xs, mean_ys, std_xs, std_ys

def plot_trajectories_rl_style(trajectories, goal: tuple[float, float], use_carpet_coords=False, 
                              show_individual=True, confidence_alpha=0.3, interpolation_points=100, color="green"):
    """
    Plot multiple trajectories in RL style with mean trajectory and confidence bands
    
    Args:
        trajectories: List of trajectory dictionaries
        goal: Goal coordinates (x, y)
        use_carpet_coords: Whether to use carpet coordinates or GPS
        show_individual: Whether to show individual trajectories (faded)
        confidence_alpha: Alpha for confidence band fill
        interpolation_points: Number of points for trajectory interpolation
    """

    mean_colors = {
        "blue": "darkblue",
        "orange": "darkorange",
        "green": "darkgreen",
        "red": "darkred"
    }
    
    # Determine coordinate system
    if use_carpet_coords:
        x_label, y_label = 'X (m)', 'Y (m)'
        traj_title = 'RL Training Trajectories - Carpet Coordinates'
        coord_keys = ('carpet_x', 'carpet_y')
    else:
        x_label, y_label = 'Longitude', 'Latitude'
        traj_title = 'RL Training Trajectories - GPS Coordinates'
        coord_keys = ('lon', 'lat')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    normalized_trajectories, mean_xs, mean_ys, std_xs, std_ys = get_mean_std_from_trajectories(trajectories, coord_keys, interpolation_points)
    
    # Plot individual trajectories (faded)
    if show_individual:
        for xs, ys in normalized_trajectories:
            ax.plot(xs, ys, color='lightblue', alpha=0.3, linewidth=1)
    
    # Plot confidence bands (±1 std)
    ax.fill_between(mean_xs, 
                    mean_ys - std_ys, 
                    mean_ys + std_ys, 
                    alpha=confidence_alpha, 
                    color=color, 
                    # label=f'±1σ confidence band (n={len(normalized_trajectories)})'
                    )
    
    # Plot mean trajectory
    ax.plot(mean_xs, mean_ys, color=mean_colors[color], linewidth=3, label='Mean trajectory')
    # ax.plot(mean_xs, mean_ys, color=mean_colors[color], linewidth=3, label='Mean trajectory', linestyle='--')
    
    # Plot start and end points of mean trajectory
    ax.scatter(mean_xs[0], mean_ys[0], color='darkgreen', s=150, marker='o', 
               edgecolors='black', linewidth=2, label='Start', zorder=5)
    ax.scatter(mean_xs[-1], mean_ys[-1], color='darkred', s=150, marker='s', 
               edgecolors='black', linewidth=2, label='Mean end', zorder=5)
    
    # Plot goal points
    ax.scatter(goal[0], goal[1], color='gold', s=200, marker='x', 
               label='Car', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(goal[0] - 0.95, goal[1], color='magenta', s=200, marker='*', 
               label='Goal', edgecolors='black', linewidth=2, zorder=5)
    
    # Calculate and display statistics
    final_distances = [np.sqrt((xs[-1] - (goal[0] - 0.95))**2 + (ys[-1] - goal[1])**2) 
                      for xs, ys in normalized_trajectories]
    mean_final_distance = np.mean(final_distances)
    std_final_distance = np.std(final_distances)
    
    # stats_text = f'Final distance to goal:\nMean: {mean_final_distance:.3f}\nStd: {std_final_distance:.3f}'
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.8))
    
    ax.set_title(traj_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add carpet boundaries if using carpet coordinates
    if use_carpet_coords:
        ax.add_patch(plt.Rectangle((0, 0), 4, 5, fill=False, edgecolor='black', 
                                  linewidth=2, linestyle='--', label='Carpet boundary'))
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 6.5)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def process_multiple_metadata_files(path_runs, goal, carpet_start):
    base_path = Path(path_runs)

    results_path = base_path / "results"
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return
    
    use_carpet_coords = carpet_start is not None

    trajectories = []
    
    # Iterate through each folder in results
    for folder in tqdm(list(results_path.iterdir())):
        if folder.is_dir():
            metadata_path = folder / "metadata.json"

            data = load_drone_data(metadata_path)

            trajectory, _ = analyze_trajectory(data, carpet_start)
            trajectories.append(trajectory)
    
    # plot_multiple_trajectories(trajectories, goal, use_carpet_coords)
    plot_trajectories_rl_style(trajectories, goal, use_carpet_coords, show_individual=False)

## ----------------------------------------------------------------------------------------------------------------------

## --------------------------------------------
## Trajectories for multiple runs
## --------------------------------------------

def plot_multiple_trajectories_rl_style(config_dicts, goal: tuple[float, float], use_carpet_coords=False, 
                              show_individual=True, confidence_alpha=0.3, interpolation_points=100,
                              goal_image_path=Paths.CAR_GOAL_PLOT,
                              goal_image_size=0.8):
    """
    Plot multiple trajectories in RL style with mean trajectory and confidence bands
    
    Args:
        trajectories: List of trajectory dictionaries
        goal: Goal coordinates (x, y)
        use_carpet_coords: Whether to use carpet coordinates or GPS
        show_individual: Whether to show individual trajectories (faded)
        confidence_alpha: Alpha for confidence band fill
        interpolation_points: Number of points for trajectory interpolation
    """

    mean_colors = {
        "blue": "darkblue",
        "orange": "darkorange",
        "green": "darkgreen",
        "red": "darkred"
    }

    mean_colors_rgb = {
        (250, 128, 114, 0): (175, 90, 80, 0), # Medium purple
        (32, 178, 170, 0): (22, 125, 119, 0), # Light sea green
        (255, 215, 0, 0): (179, 151, 0, 0), # gold
        (100, 149, 237, 0): (70, 104, 166, 0) # cornflower blue
    }
    
    # Determine coordinate system
    if use_carpet_coords:
        x_label, y_label = 'X (m)', 'Y (m)'
        traj_title = 'Sim Trajectories'
        coord_keys = ('carpet_x', 'carpet_y')
    else:
        x_label, y_label = 'Longitude', 'Latitude'
        traj_title = 'Sim Trajectories'
        coord_keys = ('lon', 'lat')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for config in config_dicts:
        trajectories = config["trajectories"]
        
        color = config["color"]
        if color in mean_colors:
            mean_color = mean_colors[color]
        else:
            mean_color = mean_colors_rgb[color]

            print(f"{color=}")

            color = (color[0]/255, color[1]/255, color[2]/255)
            mean_color = (mean_color[0]/255, mean_color[1]/255, mean_color[2]/255)
        
        name = config["name"]

        normalized_trajectories, mean_xs, mean_ys, std_xs, std_ys = get_mean_std_from_trajectories(trajectories, coord_keys, interpolation_points)
        
        # Plot individual trajectories (faded)
        if show_individual:
            for xs, ys in normalized_trajectories:
                ax.plot(xs, ys, color='lightblue', alpha=0.3, linewidth=1)
        
        # Plot confidence bands (±1 std)
        ax.fill_between(mean_xs, 
                        mean_ys - std_ys, 
                        mean_ys + std_ys, 
                        alpha=confidence_alpha, 
                        color=color, 
                        # label=f'±1σ confidence band (n={len(normalized_trajectories)})'
                        )
        
        # Plot mean trajectory
        if ("dotted" in config):
            ax.plot(mean_xs, mean_ys, color=mean_color, linewidth=3, label=f'Mean trajectory - {name}', linestyle='--')
        else:
            ax.plot(mean_xs, mean_ys, color=mean_color, linewidth=3, label=f'Mean trajectory - {name}')
        
        # Plot start point
        # ax.scatter(mean_xs[0], mean_ys[0], color='darkgreen', s=150, marker='o', 
        #             edgecolors='black', linewidth=2, label='Start', zorder=5)
        ax.scatter(mean_xs[0], mean_ys[0], color='darkgreen', s=150, marker='o', 
                    edgecolors='black', linewidth=2, zorder=5)
        
        # Plot end points of mean trajectory
        # if ("dotted" in config):
        #     ax.scatter(mean_xs[-1], mean_ys[-1], color='darkred', s=150, marker='s', 
        #         edgecolors='black', linewidth=2, label=f'Mean end - {name}', linestyle='--', zorder=5)
        # else:
        #     ax.scatter(mean_xs[-1], mean_ys[-1], color='darkred', s=150, marker='s', 
        #         edgecolors='black', linewidth=2, label=f'Mean end - {name}', zorder=5)
        
        # Calculate and display statistics
        final_distances = [np.sqrt((xs[-1] - (goal[0] - 0.95))**2 + (ys[-1] - goal[1])**2) 
                        for xs, ys in normalized_trajectories]
        mean_final_distance = np.mean(final_distances)
        std_final_distance = np.std(final_distances)
        
    # stats_text = f'Final distance to goal:\nMean: {mean_final_distance:.3f}\nStd: {std_final_distance:.3f}'
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.8))

    # Plot goal points
    ax.scatter(goal[0] - 0.95, goal[1], color='magenta', s=200, marker='*', 
                label=f'Goal', edgecolors='black', linewidth=2, zorder=5)

    
    # ax.scatter(goal[0], goal[1], color='gold', s=200, marker='x', 
    #             label='Car', edgecolors='black', linewidth=2, zorder=5)
    # Plot goal with image or default marker
    if goal_image_path:
        try:
            img = mpimg.imread(goal_image_path)
            
            # Calculate zoom to make image width = 0.3 meters
            # Get current axis limits to calculate pixels per data unit
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            fig_width, fig_height = fig.get_size_inches()
            
            # Calculate data units per inch
            data_width = xlim[1] - xlim[0]
            data_height = ylim[1] - ylim[0]
            
            # Calculate pixels per data unit (approximate)
            dpi = fig.dpi
            pixels_per_inch_x = (fig_width * dpi) / data_width
            
            # Calculate zoom factor to make image width = 0.3 meters
            desired_width_meters = 0.25
            desired_width_pixels = desired_width_meters * pixels_per_inch_x
            current_width_pixels = img.shape[1]  # Image width in pixels
            zoom_factor = desired_width_pixels / current_width_pixels
            
            imagebox = OffsetImage(img, zoom=zoom_factor)
            ab = AnnotationBbox(imagebox, (goal[0], goal[1]), frameon=False, zorder=10)
            ax.add_artist(ab)
            
            # Add invisible scatter point for legend
            # ax.scatter(goal[0], goal[1], color='gold', s=200, marker='s', 
            #           label='Car', edgecolors='black', linewidth=2, zorder=100)
        except Exception as e:
            print(f"Could not load image '{goal_image_path}': {e}")
            print("Using default marker instead.")
            # Fallback to default marker
            ax.scatter(goal[0], goal[1], color='gold', s=200, marker='x', 
                      label='Car', edgecolors='black', linewidth=2, zorder=5)
    else:
        # Use default marker
        ax.scatter(goal[0], goal[1], color='gold', s=200, marker='x', 
                  label='Car', edgecolors='black', linewidth=2, zorder=5)

    # ax.set_title(traj_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.legend(loc='upper right')
    ax.legend(loc='upper right', fontsize='x-large')
    ax.grid(True, alpha=0.3)
    
    # Add carpet boundaries if using carpet coordinates
    if use_carpet_coords:
        ax.add_patch(plt.Rectangle((0, 0), 4, 5, fill=False, edgecolor='black', 
                                  linewidth=2, linestyle='--', label='Carpet boundary'))
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def process_comparison_multiple_metadata_files(paths_runs_configs, goal):
    config_dicts = []

    use_carpet_coords = True

    for path_runs_config_name, path_runs_config in paths_runs_configs.items():
        path_runs = path_runs_config["path"]
        base_path = Path(path_runs)

        config_dict = deepcopy(path_runs_config)
        config_dict["name"] = path_runs_config_name

        results_path = base_path / "results"
        
        if not results_path.exists():
            print(f"Results directory not found: {results_path}")
            return
        
        carpet_start = path_runs_config["carpet_start"]
        use_carpet_coords = carpet_start is not None

        trajectories = []
        
        # Iterate through each folder in results
        for folder in tqdm(list(results_path.iterdir())):
            if folder.is_dir():
                metadata_path = folder / "metadata.json"
                if (not metadata_path.exists()):
                    continue

                data = load_drone_data(metadata_path)

                trajectory, _ = analyze_trajectory(data, carpet_start)
                trajectories.append(trajectory)
        
        config_dict["trajectories"] = trajectories
        config_dicts.append(config_dict)
        
    
    plot_multiple_trajectories_rl_style(config_dicts, goal, use_carpet_coords, show_individual=False)


## --------------------------------------------
## --------------------------------------------

if __name__ == "__main__":
    path = "/home/brittle/Desktop/work/code/space-time-lab-org/auto-follow/output/results_without_frames_poli_pc/bunker-online-4k-config-test-down-left/results/2025-06-11_23-45-17/metadata.json"
    path = "/home/brittle/Desktop/work/code/space-time-lab-org/auto-follow/output/results_without_frames_poli_pc/bunker-online-4k-config-test-front-small-offset-left/results/2025-06-12_06-15-25/metadata.json"
    path = "/home/brittle/Desktop/work/code/space-time-lab-org/drone-base/examples/results/2025-06-12_18-07-21/metadata.json"
    path = "/home/brittle/Desktop/work/code/space-time-lab-org/drone-base/examples/results/2025-06-12_18-21-51/metadata.json"
    path = "/home/brittle/Downloads/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/results-real-ibvs-all/real-ibvs-down-left/results/2025-06-15_19-22-10/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged/real-ibvs-up-left/results/2025-06-13_20-36-19/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/ibvs/sim-car-ibvs-results-poli-pc/bunker-online-4k-config-test-front-small-offset-right/results/2025-06-15_05-39-12/metadata.json"
    path = "/home/brittle/Desktop/work/space-time-vision-repos/auto-follow/output/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-23-36/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged/real-ibvs-front-small-offset-left/results/2025-06-15_18-39-47/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student/sim-student-results-pc-sebnae/bunker-online-4k-config-test-front-small-offset-right-student/results/2025-06-15_18-19-01/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student/sim-car-student-results-poli-pc/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_04-00-25/metadata.json"
    path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student/sim-student-results-pc-sebi/bunker-online-4k-config-test-front-small-offset-left-student/results/2025-06-16_02-01-23/metadata.json"

    # FRONT RIGHT ON STUDENT
    # path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student/sim-student-results-pc-sebi/bunker-online-4k-config-test-front-small-offset-right-student/results/2025-06-16_01-44-34/metadata.json"

    ## TODO
    ## - plot the jerk (second derivative of speed (?))
    ## - plot trajectory
    ## - plot speed vs time

    '''
    TODO paper
    - mention yaw clipping (due to segmentation) for IBVS // experiments / method
    TLDR
    keywords - in abstract
    '''

    goal = (2, 2.5)

    ## -------------------------------------------------

    ## teacher; student

    ## front-small-offset-right
    paths = {
        # "ibvs-front-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/real/real-world-ibvs-results-merged/real-ibvs-front-small-offset-right",
        #     "color": "blue",
        #     "carpet_start": (2 + 1.5, 2.5 - 0.3)
        # },
        # "student-front-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/real-student-21-06/real-student-front-small-offset-right",
        #     "color": "orange",
        #     "dotted": True,
        #     "carpet_start": (2 + 1.5, 2.5 - 0.3)
        # },

        # --------------------------------------------------

        "Teacher front-right": {
            "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-front-small-offset-right",
            "color": "blue",
            "carpet_start": (2 + 1.5, 2.5 - 0.3)
        },
        "Student front-right": {
            "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-front-small-offset-right-student",
            "color": "orange",
            "dotted": True,
            "carpet_start": (2 + 1.5, 2.5 - 0.3)
        },

        "Teacher front-left": {
            "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-front-small-offset-left",
            "color": "green",
            "carpet_start": (2 + 1.5, 2.5 + 0.3)
        },
        "Student front-left": {
            "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-front-small-offset-left-student",
            "color": "red",
            "dotted": True,
            "carpet_start": (2 + 1.5, 2.5 + 0.3)
        },

        # ------------------------------------

        # "ibvs-left": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-left",
        #     "color": (250, 128, 114, 0),
        #     "carpet_start": (2 + 0, 2.5 + 1)
        # },
        # "student-left": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-left-student",
        #     "color": (100, 149, 237, 0),
        #     "dotted": True,
        #     "carpet_start": (2 + 0, 2.5 + 1)
        # },

        # "ibvs-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-right",
        #     "color": (255, 215, 0, 0),
        #     "carpet_start": (2 + 0, 2.5 - 1)
        # },
        # "student-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-right-student",
        #     "color": (32, 178, 170, 0),
        #     "dotted": True,
        #     "carpet_start": (2 + 0, 2.5 - 1)
        # },

        # ------------------------------------

        # "ibvs-up-left": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-up-left",
        #     "color": (250, 128, 114, 0),
        #     "carpet_start": (2 + 1, 2.5 + 1)
        # },
        # "student-up-left": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-up-left-student",
        #     "color": (100, 149, 237, 0),
        #     "dotted": True,
        #     "carpet_start": (2 + 1, 2.5 + 1)
        # },

        # "ibvs-up-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-up-right",
        #     "color": (255, 215, 0, 0),
        #     "carpet_start": (2 + 1, 2.5 - 1)
        # },
        # "student-up-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-up-right-student",
        #     "color": (32, 178, 170, 0),
        #     "dotted": True,
        #     "carpet_start": (2 + 1, 2.5 - 1)
        # },

        # ----------------------------------------

        # "ibvs-down-left": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-down-left",
        #     "color": (250, 128, 114, 0),
        #     "carpet_start": (2 - 1, 2.5 + 1)
        # },
        # "student-down-left": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-down-left-student",
        #     "color": (100, 149, 237, 0),
        #     "dotted": True,
        #     "carpet_start": (2 - 1, 2.5 + 1)
        # },

        # "ibvs-down-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-ibvs-results-merged/bunker-online-4k-config-test-down-right",
        #     "color": (255, 215, 0, 0),
        #     "carpet_start": (2 - 1, 2.5 - 1)
        # },
        # "student-down-right": {
        #     "path": "/media/mihaib08/0AC68039C68026D3/models/_research_drone/output/droid-data/sim/sim-student-results-merged/bunker-online-4k-config-test-down-right-student",
        #     "color": (32, 178, 170, 0),
        #     "dotted": True,
        #     "carpet_start": (2 - 1, 2.5 - 1)
        # },
    }

    process_comparison_multiple_metadata_files(paths, goal=goal)
