#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly RUN_ENV_SCRIPT_PATH="${SCRIPT_DIR}/../assets/environment/run_environment_car.sh"

# Check for environment variable or command line argument
if [[ $# -ge 1 && "$1" != -* ]]; then
    COMMAND="$1"
    shift
elif [[ -n "${BUNKER_APP_PATH:-}" ]]; then
    COMMAND="${BUNKER_APP_PATH}"
else
    echo "Error: Bunker application path not specified."
    echo ""
    echo "Usage: $0 <path-to-UnrealApp.sh> [additional-options]"
    echo "   or: export BUNKER_APP_PATH=/path/to/UnrealApp.sh"
    echo ""
    echo "Example:"
    echo "  $0 /home/user/Games/DroneSimulation/bunker/installed/LinuxNoEditor/UnrealApp.sh"
    echo "  BUNKER_APP_PATH=/path/to/UnrealApp.sh $0"
    exit 1
fi

# Verify the path exists
if [[ ! -x "${COMMAND}" ]]; then
    echo "Error: '${COMMAND}' does not exist or is not executable."
    exit 1
fi

echo "Using Parrot Sphinx UE4 app: ${COMMAND}"

"${RUN_ENV_SCRIPT_PATH}" -command="${COMMAND}" -level=main_details -config=add_mc_laren -quality=low
