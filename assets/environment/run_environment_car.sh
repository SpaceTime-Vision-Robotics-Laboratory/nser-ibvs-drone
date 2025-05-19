#!/usr/bin/env bash
set -euo pipefail

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

for cmd in sed find parrot-ue4-empty; do
    if ! command_exists "$cmd"; then
        echo >&2 "$cmd is required but not installed. Aborting."
        exit 1
    fi
done

readonly BUNKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly CONFIG_DIR="${BUNKER_DIR}/config"
readonly DEFAULT_CONFIG_NAME="bunker_with_car"
readonly DEFAULT_CMD="parrot-ue4-empty"
readonly DEFAULT_LEVEL="main"

list_configs() {
    find "${CONFIG_DIR}" -maxdepth 1 -name "*.yaml" -exec basename {} .yaml \;
}

CONFIG_NAME="${DEFAULT_CONFIG_NAME}"
QUALITY="high"
RENDER_OFF_SCREEN=false
COMMAND="${DEFAULT_CMD}"
LEVEL="${DEFAULT_LEVEL}"

while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

readonly ORIGINAL_CONFIG="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
if [[ ! -f "${ORIGINAL_CONFIG}" ]]; then
    echo "Error: Config file does not exist: ${ORIGINAL_CONFIG}"
    echo "Available configs are:"
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