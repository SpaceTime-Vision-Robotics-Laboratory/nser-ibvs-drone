# Bunker UE4 Environment

This directory contains the configuration system for launching Parrot Sphinx
with custom Unreal Engine environments. The main script handles model path resolution,
configuration file processing, and cleanup.

## Prerequisites

- Parrot Sphinx simulator installed (parrot-ue4-empty available in PATH)
- Built UE4 environment (optional, for custom environments)

To replicate the experiments use the bunker built UE4 environment.

## Directory Structure

```bash
environment/
├── run_environment_car.sh    # Main launcher script
├── config/                   # YAML configuration files
│   ├── bunker_with_car.yaml  # Default: bunker + car
│   ├── add_mc_laren.yaml     # McLaren vehicle injection
│   └── ...
├── models/                   # 3D model files (.fbx)
└── README.md
```

## Usage:

```bash
# Run with all defaults (parrot-ue4-empty, main level, bunker_with_car config, high quality)
./run_environment_car.sh

# Run custom UE4 bunker build with car on main_details level
./run_environment_car.sh \
    -command=/path/to/UnrealApp.sh \
    -level=main_details \
    -config=add_mc_laren \
    -quality=low
```

### Options

| Option             | Default            | Description                                                   |
|--------------------|--------------------|---------------------------------------------------------------|
| `-command=<path>`  | `parrot-ue4-empty` | Path to UE4 executable or Parrot environment                  |
| `-level=<name>`    | `main`             | UE4 level to load (see [Available Levels](#available-levels)) |
| `-config=<name>`   | `bunker_with_car`  | Config file name without `.yaml` extension                    |
| `-quality=<level>` | `high`             | Rendering quality: `low` or `high`                            |
| `-RenderOffScreen` | disabled           | Run without display (headless mode)                           |
| `--list-envs`      | -                  | List available configuration files                            |
| `-h, --help`       | -                  | Show help message                                             |

## Available Levels

Important levels for this repository:

| Level             | Description                                                        | Drone Spawn Point |
|-------------------|--------------------------------------------------------------------|-------------------|
| `main`            | Bunker with carpet and lights                                      | (0, 0, 30)        |
| `main_details`    | Enhanced bunker with details (doors, handles, sockets, exit signs) | (-153.75, 0, 30)  |
| `main_center_car` | Like `main_details` but with pre-spawned car at center             | (0, 0, 12)        |

## Configuration Files

Configuration files define what objects to spawn in the environment. They use YAML format and support
the `${MODELS_DIR}` placeholder which gets resolved to the absolute path of the [models/](./models) directory.

List available configs:

```bash
./run_environment_car.sh --list-envs
```

### Common Configurations

- `bunker_with_car`: Default config with bunker (without details), carpet (2K resolution) and car models
- `add_mc_laren`: Injects vehicle into the scene in the center of the carpet

## How It Works

1. **Validates dependencies:** Checks for required commands (`sed`, `find`, `parrot-ue4-empty`)
2. **Resolves model paths:** Exports `BUNKER_MODELS_DIR` and creates a temporary config with absolute paths
3. **Launches environment:** Runs the specified UE4 command with all parameters
4. **Cleanup:** Removes temporary config file on exit (via trap)

## Examples

### Basic: Empty Environment with Default Config

```bash
./run_environment_car.sh
```

### Custom Build with Detailed Bunker

```bash
./run_environment_car.sh \
    -command=/home/brittle/Games/MyGames/DroneSimulation/bunker/installed/LinuxNoEditor/UnrealApp.sh \
    -level=main_details \
    -config=add_mc_laren \
    -quality=low
```

### Headless Mode

```bash
./run_environment_car.sh \
    -level=main \
    -quality=low \
    -RenderOffScreen
```

### Direct Parrot Sphinx UE4 Launch (bypassing this script)

If you want to run the UE4 environment directly without dynamic object injection:

```bash
/path/to/installed/LinuxNoEditor/UnrealApp.sh -level=main_center_car
```

## Related Scripts

The wrapper scripts in `scripts/` provide simplified interfaces to this configuration system:

- [start_bunker_env.sh](../../scripts/start_bunker_env.sh): Launches custom-built bunker environment with car injection
- [start_empty_bunker_env.sh](../../scripts/start_empty_bunker_env.sh): Launches `parrot-ue4-empty` with bunker, car and
  carpet dynamic injection

## Troubleshooting

### "parrot-ue4-empty is required but not installed"

Ensure Parrot Sphinx is installed and its binaries are in your PATH.

### "Config file does not exist"

Run `./run_environment_car.sh --list-envs` to see available configs. Config files must be in the `config/` directory.

### Environment crashes on startup

Try running with `-quality=low` to reduce GPU requirements.

Try restarting the firmwared service which solves a common error when simulator crashes or closes unexpectedly:

```bash
sudo systemctl restart firmwared.service
```
