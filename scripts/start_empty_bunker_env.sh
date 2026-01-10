#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly RUN_ENV_SCRIPT_PATH="${SCRIPT_DIR}/../assets/environment/run_environment_car.sh"

"${RUN_ENV_SCRIPT_PATH}" -command=parrot-ue4-empty -level=main -quality=low
