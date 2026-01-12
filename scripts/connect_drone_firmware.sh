#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_DRONE_PATH="/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"
readonly DEFAULT_FIRMWARE="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"

declare -A POSES=(
    ["default"]="-1.5375 0 0.3 0 0 0"
    ["down-left"]="-1 1 0.3 0 -0 5.495"
    ["down-right"]="-1 -1 0.3 0 -0 0.785"
    ["front-small-offset-left"]="1.5 0.3 0.3 0 -0 -2.9"
    ["front-small-offset-right"]="1.5 -0.3 0.3 0 -0 2.9"
    ["left"]="0 1 0.3 0 -0 -1.57"
    ["right"]="0 -1 0.3 0 -0 1.57"
    ["up-left"]="1 1 0.3 0 -0 3.925"
    ["up-right"]="1 -1 0.3 0 -0 2.355"
)


usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Start Parrot Sphinx with an Anafi drone at a specified pose.

Options:
    -p, --pose <name>       Pose configuration name (default: default)
    -c, --custom <pose>     Custom pose string "x y z roll pitch yaw"
    -l, --list              List available pose configurations
    -h, --help              Show this help message

Available poses:
EOF
    for pose_name in "${!POSES[@]}"; do
        printf "    %-28s %s\n" "${pose_name}" "${POSES[${pose_name}]}"
    done | sort
}

list_poses() {
    echo "Available pose configurations:"
    echo ""
    printf "%-30s %s\n" "NAME" "POSE (x y z roll pitch yaw)"
    printf "%-30s %s\n" "----" "--------------------------"
    for pose_name in "${!POSES[@]}"; do
        printf "%-30s %s\n" "${pose_name}" "${POSES[${pose_name}]}"
    done | sort
}

POSE_NAME="default"
CUSTOM_POSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pose)
            shift
            POSE_NAME="$1"
            ;;
        -p=*|--pose=*)
            POSE_NAME="${1#*=}"
            ;;
        -c|--custom)
            shift
            CUSTOM_POSE="$1"
            ;;
        -c=*|--custom=*)
            CUSTOM_POSE="${1#*=}"
            ;;
        -l|--list)
            list_poses
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ -n "${CUSTOM_POSE}" ]]; then
    POSE="${CUSTOM_POSE}"
    echo "Using custom pose: ${POSE}"
elif [[ -v "POSES[${POSE_NAME}]" ]]; then
    POSE="${POSES[${POSE_NAME}]}"
    echo "Using pose '${POSE_NAME}': ${POSE}"
else
    echo "Error: Unknown pose '${POSE_NAME}'"
    echo ""
    echo "Available poses:"
    for pose_name in "${!POSES[@]}"; do
        echo "  - ${pose_name}"
    done | sort
    exit 1
fi

DRONE_CONFIG="${DEFAULT_DRONE_PATH}::pose=${POSE}::firmware=${DEFAULT_FIRMWARE}"

echo "Starting Sphinx with Anafi drone..."
echo "Drone config: ${DRONE_CONFIG}"
echo ""

exec sphinx "${DRONE_CONFIG}"
