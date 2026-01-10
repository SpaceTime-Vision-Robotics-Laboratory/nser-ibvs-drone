# Simulation Launch Scripts

This directory contains shell scripts for launching the Parrot Sphinx drone simulator with various
environment configurations and connecting to drone firmware.

## Prerequisites

- Parrot Sphinx simulator installed
- `gnome-terminal` (for [start_carla_env.sh](start_carla_env.sh))
- Access to the Unreal Engine environments (bunker/CARLA)

## Directory Structure:

```bash
scripts/
├── connect_drone_firmware.sh   # Drone firmware connection with pose options
├── start_bunker_env.sh         # Bunker environment with vehicle injection
├── start_carla_env.sh          # parrot-ue4-carla - Full CARLA simulation setup
├── start_empty_bunker_env.sh   # Empty bunker environment with simple bunker and car injection
└── README.md                   
```

## Typical Workflow

1. Start the desired environment script ([start_bunker_env.sh](start_bunker_env.sh),
   [start_carla_env.sh](start_carla_env.sh), or [start_empty_bunker_env.sh](start_empty_bunker_env.sh))
2. If not using [start_carla_env.sh](start_carla_env.sh), run [connect_drone_firmware.sh](connect_drone_firmware.sh)
   separately to spawn the drone
3. Run your application code to control the drone

## Scripts Description

### [connect_drone_firmware.sh](connect_drone_firmware.sh)

Starts Parrot Sphinx with an Anafi 4K drone at a specified pose position.

```bash
# Use default pose
./connect_drone_firmware.sh

# Use a named pose
./connect_drone_firmware.sh --pose=left

# Use a custom pose (x y z roll pitch yaw)
./connect_drone_firmware.sh --custom "1.0 0.5 0.3 0 0 1.57"

# List all available poses
./connect_drone_firmware.sh --list

# See available script documentation
./connect_drone_firmware.sh --help
```

#### Available Poses:

| Name                       | Position (x y z roll pitch yaw) |
|----------------------------|---------------------------------|
| `default`                  | -1.5375 0 0.3 0 0 0             |
| `down-left`                | -1 1 0.3 0 0 5.495              |
| `down-right`               | -1 -1 0.3 0 0 0.785             |
| `front-small-offset-left`  | 1.5 0.3 0.3 0 0 -2.9            |
| `front-small-offset-right` | 1.5 -0.3 0.3 0 0 2.9            |
| `left`                     | 0 1 0.3 0 0 -1.57               |
| `right`                    | 0 -1 0.3 0 0 1.57               |
| `up-left`                  | 1 1 0.3 0 0 3.925               |
| `up-right`                 | 1 -1 0.3 0 0 2.355              |

### [start_bunker_env.sh](start_bunker_env.sh)

Launches the custom-built Unreal Engine bunker environment for Parrot Sphinx with a vehicle configuration.

```bash
# Provide path as argument
./start_bunker_env.sh /path/to/UnrealApp.sh

# Or set environment variable
export BUNKER_APP_PATH=/path/to/UnrealApp.sh
./start_bunker_env.sh
```

You can edit your `.bashrc` for persistent `BUNKER_APP_PATH` environment variable

**Configuration:** Runs with `main_details` level, `add_mc_laren` config, and `low` quality settings.

### [start_empty_bunker_env.sh](start_empty_bunker_env.sh)

Launches an empty bunker environment using `parrot-ue4-empty` and populating it with `.fbx` files
(i.e., simple bunker mesh without details carpet texture with 2K resolution and the car).

```
./start_empty_bunker_env.sh
```

**Configuration:** Runs with `main` level and `low` quality settings.

### [start_carla_env.sh](start_carla_env.sh)

Launches a full `parrot-ue4-carla` CARLA simulation environment by opening three terminal windows:

1. **CARLA UE4:** Starts the CARLA environment with Town 10
2. **Sphinx:** Connects the Anafi drone with firmware at the "Pickup" pose
3. **Sphinx CLI:** Unpauses the simulation after 10 seconds

```bash
./start_carla_env.sh
```

> **Note:** This script requires `gnome-terminal` and will open three separate terminal windows.


## Dependencies
These scripts rely on the environment configuration system located in `../assets/environment/`. 
See [run_environment_car.sh](../assets/environment/run_environment_car.sh) for available 
configuration options (levels, quality settings, configs).