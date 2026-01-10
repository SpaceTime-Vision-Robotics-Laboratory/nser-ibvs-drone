#!/usr/bin/env bash
set -euo pipefail

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

readonly SCRIPT_NAME="$(basename "$0")"
readonly BUNKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly CONFIG_DIR="${BUNKER_DIR}/config"
readonly DEFAULT_CONFIG_NAME="bunker_with_car"
readonly DEFAULT_CMD="parrot-ue4-empty"
readonly DEFAULT_LEVEL="main"
readonly DEFAULT_QUALITY="high"

list_configs() {
    echo "Available configurations:"
    find "${CONFIG_DIR}" -maxdepth 1 -name "*.yaml" -exec basename {} .yaml \; | sort | while read -r config; do
        echo "  - ${config}"
    done
}


usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Launch Parrot Sphinx with a configured Unreal Engine environment.

This script handles model path resolution, creates a temporary config with
absolute paths, and cleans up on exit.

Options:
    -command=<path>       UE4 executable or Parrot environment (default: ${DEFAULT_CMD})
    -level=<name>         UE4 level to load (default: ${DEFAULT_LEVEL})
    -config=<name>        Config file name without .yaml extension (default: ${DEFAULT_CONFIG_NAME})
    -quality=<level>      Rendering quality: low or high (default: ${DEFAULT_QUALITY})
    -RenderOffScreen      Run in headless mode without display
    --list-envs           List available configuration files
    -h, --help            Show this help message

Available levels:
    main                  Bunker with carpet and lights, spawn at (0, 0, 30)
    main_details          Enhanced bunker with details, spawn at (-153.75, 0, 30)
    main_center_car       Like main_details with pre-spawned car at (0, 0, 12)

Examples:
    # Run with defaults
    ${SCRIPT_NAME}

    # Custom build with car
    ${SCRIPT_NAME} -command=/path/to/UnrealApp.sh -level=main_details -config=add_mc_laren -quality=low

    # Headless mode
    ${SCRIPT_NAME} -level=main -quality=low -RenderOffScreen

EOF
}

for cmd in sed find parrot-ue4-empty; do
    if ! command_exists "$cmd"; then
        echo >&2 "Error: ${cmd} is required but not installed. Aborting."
        exit 1
    fi
done

CONFIG_NAME="${DEFAULT_CONFIG_NAME}"
QUALITY="${DEFAULT_QUALITY}"
RENDER_OFF_SCREEN=false
COMMAND="${DEFAULT_CMD}"
LEVEL="${DEFAULT_LEVEL}"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --list-envs)
            list_configs
            exit 0
            ;;
        -config=*|--config=*)
            CONFIG_NAME="${1#*=}"
            ;;
        -config|--config)
            shift
            CONFIG_NAME="$2"
            ;;
        -quality=*|--quality=*)
            QUALITY="${1#*=}"
            ;;
        -quality|--quality)
            shift
            QUALITY="$2"
            ;;
        -command=*|--command=*)
            COMMAND="${1#*=}"
            ;;
        -command|--command)
            shift
            COMMAND="$1"
            ;;
        -level=*|--level=*)
            LEVEL="${1#*=}"
            ;;
        -level|--level)
            shift
            LEVEL="$1"
            ;;
        -RenderOffScreen|--RenderOffScreen)
            RENDER_OFF_SCREEN=true
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
    shift
done

readonly ORIGINAL_CONFIG="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
if [[ ! -f "${ORIGINAL_CONFIG}" ]]; then
    echo "Error: Config file does not exist: ${ORIGINAL_CONFIG}"
    list_configs
    exit 1
fi
echo "Using config file: ${ORIGINAL_CONFIG}"

export BUNKER_MODELS_DIR="${BUNKER_DIR}/models"
echo "Exported bunker models path to: ${BUNKER_MODELS_DIR}"

readonly TEMP_CONFIG="${BUNKER_DIR}/config/temp_config.yaml"
sed "s|\${MODELS_DIR}|${BUNKER_MODELS_DIR}|g" "$ORIGINAL_CONFIG" > "$TEMP_CONFIG"
echo "Created temporary config file that contains absolute paths: ${TEMP_CONFIG}"
echo "Contents of temporary config file:"
echo  "--------------------------------"
cat "${TEMP_CONFIG}"
printf "\n--------------------------------\n"

cleanup() {
    echo "Terminating program. Removing temporary files..."
    [[ -f "${TEMP_CONFIG}" ]] && rm "${TEMP_CONFIG}"
    echo "Temporary files removed. Exiting."
}
trap cleanup EXIT

readonly PROJECT_ROOT="$( cd "${BUNKER_DIR}/../.." && pwd )"
echo "Moving to project root: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

PARROT_CMD=(
    "${COMMAND}" "-level=${LEVEL}"
    "-config-file=${TEMP_CONFIG}"
    "-quality=${QUALITY}"
)
  [[ "${RENDER_OFF_SCREEN}" == true ]] && PARROT_CMD+=(-RenderOffScreen)

echo "Environment configuration script completed. Running Parrot Sphinx Bunker environment..."
echo "Executing command: ${PARROT_CMD[*]}"
"${PARROT_CMD[@]}"