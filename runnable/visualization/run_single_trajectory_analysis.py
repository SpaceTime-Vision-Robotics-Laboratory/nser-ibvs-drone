import argparse

from nser_ibvs_drone.utils.path_manager import Paths
from nser_ibvs_drone.visualization.metadata_trajectory_analysis import get_carpet_start, trajectory_analysis_single_run


def parse_tuple(value):
    """Parse a comma-separated string into a tuple of floats."""
    try:
        x, y = value.split(',')
        return float(x.strip()), float(y.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: '{value}'. Expected 'x,y' (e.g., '2.0,2.5')")


def main():
    """
    Possible directions for the current configuration:
        - "front-right", "front-left"
        - "left", "right"
        - "up-left", "up-right"
        - "down-left", "down-right"
        - Custom: "x,y" offset (e.g., "1.5,0.3")

    Example of usage:
    python runnable/visualization/run_single_trajectory_analysis.py \
        --metadata_path=/path/to/experiment/bunker-front-left/metadata.json \
        --direction=front-left \
        --save_path=/path/to/output/directory/image-name.png
    """
    save_path = Paths.BASE_DIR / "results" / "trajectories" / "individual" / "single-trajectory.png"
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, help="Path to metadata.json that have the drone logs")
    parser.add_argument("--direction", type=str, help="Direction relative to goal (predefined name or 'x,y' offset)")
    parser.add_argument("--goal", type=parse_tuple, default=(2.0, 2.5), help="Goal position as 'x,y' (default: 2.0,2.5)")
    parser.add_argument("--save_path", type=str, default=save_path, help="Path to save the result")
    args = parser.parse_args()

    try:
        custom_offset = parse_tuple(args.direction)
        carpet_start_coords = (args.goal[0] + custom_offset[0], args.goal[1] + custom_offset[1])
    except argparse.ArgumentTypeError:
        carpet_start_coords = get_carpet_start(args.goal, args.direction)
        if carpet_start_coords is None:
            return

    trajectory_analysis_single_run(
        args.metadata_path, carpet_start=carpet_start_coords, goal=args.goal, save_path=args.save_path
    )


if __name__ == '__main__':
    main()
