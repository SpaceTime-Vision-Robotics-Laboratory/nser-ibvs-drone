import json
import math

import matplotlib.pyplot as plt


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
        print(f"Using carpet coordinates with start position: ({carpet_x:.2f}, {carpet_y:.2f}) meters")
    elif not has_gps:
        print("No GPS data found. Use --carpet-start to enable trajectory analysis.")
        return [], 0.0

    for i, point in enumerate(data):
        drone = point['drone']
        speed = drone['speed']

        traj_point = {
            'time': point['time'] - data[0]['time'],  # Relative time
            'speed': math.sqrt(speed['north'] ** 2 + speed['east'] ** 2 + speed['down'] ** 2),
            'state': drone['flying_state']
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


def main(file_path, goal: tuple[float, float], carpet_start: tuple[float, float] | None = None, plot=True, ):
    data = load_drone_data(file_path)
    print(f"Distance: {calculate_drone_distance(data)}")
    trajectory, total_distance = analyze_trajectory(data, carpet_start)

    use_carpet_coords = carpet_start is not None
    print_results(trajectory, total_distance, use_carpet_coords)

    if plot:
        plot_trajectory(trajectory, goal, use_carpet_coords)


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

    goal = (2, 2.5)
    # x_carpet = 3.5
    # y_carpet = 4.5
    x_carpet = 2 + 1.5
    y_carpet = 2.5 + 0.3
    # main(path, carpet_start=(x_carpet, y_carpet), goal=goal)

    # path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/ibvs/sim-ibvs-results-merged/bunker-online-4k-config-test-front-small-offset-right/results/2025-06-14_22-32-17/metadata.json"
    # # path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/sim/student/sim-student-results-merged/bunker-online-4k-config-test-front-small-offset-right-student/results/2025-06-16_10-04-58/metadata.json"
    # path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/ibvs/real-world-ibvs-results-merged/real-ibvs-front-small-offset-right/results/2025-06-15_18-26-17/metadata.json"
    # path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/student/results-real-student-all/real-student-front-small-offset-right/results/2025-06-15_17-02-15/metadata.json"
    # path = "/home/brittle/Desktop/work/data/car-ibvs-data-tests/real/student/results-real-student-all/real-student-front-small-offset-right/results/2025-06-15_17-02-15/metadata.json"
    main(path, carpet_start=(x_carpet, y_carpet), goal=goal)
