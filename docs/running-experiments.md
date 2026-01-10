# Running Experiments
This guide explains how to run visual servoing experiments using our framework.

## Overview
The framework supports two control methods:
1. **NSER IBVS Splitter (Teacher):** Classical visual servoing using YOLO detection + mask splitting + IBVS control law
2. **Distilled Student:** Lightweight neural network trained to mimic the teacher's behavior

Both can run in the Parrot Sphinx simulator or on real hardware.

## Prerequisites
Before running experiments, ensure you have:
- Completed the [Installation Guide](installation-guide.md)
- Parrot Sphinx simulator installed
- Pre-trained models in the [models/](../models) directory
- Downloaded the bunker custom-built UE4 environment

## Simulator Experiments

### Automated Test Suite
The [run_simulator_experiments.py](../runnable/run_simulator_experiments.py) script runs experiments across 
all 8 test scenarios automatically:
```bash
python runnable/run_simulator_experiments.py \
    --sphinx_bunker_base_dir=/path/to/UnrealApp.sh \
    --target_runs=5 \
    --is_student  # Omit for IBVS teacher
```

This will open the simulator, connect the drone firmware and run the python script referring
to teacher NSER-IBVS model if `--is_student` argument is not provided or student model otherwise. 
Each run is real-time, unfortunately they cannot be parallelized.

**Arguments:**

| Argument                   | Description                         | Default  |
|----------------------------|-------------------------------------|----------|
| `--sphinx_bunker_base_dir` | Path to the UE4 bunker application  | Required |
| `--target_runs`            | Number of runs per scenario         | 1        |
| `--is_student`             | Run student network instead of IBVS | False    |


![Scenarios starting points with goal pose.](./images/Drone-Starting-Point.png)

**Test Scenarios:**

The suite runs 8 different starting positions:

| Scenario                   | Drone Start Position | Description                    |
|----------------------------|----------------------|--------------------------------|
| `down-left`                | (-1, 1, 0.3)         | Behind-left of target          |
| `down-right`               | (-1, -1, 0.3)        | Behind-right of target         |
| `front-small-offset-left`  | (1.5, 0.3, 0.3)      | Front with slight left offset  |
| `front-small-offset-right` | (1.5, -0.3, 0.3)     | Front with slight right offset |
| `left`                     | (0, 1, 0.3)          | Direct left of target          |
| `right`                    | (0, -1, 0.3)         | Direct right of target         |
| `up-left`                  | (1, 1, 0.3)          | Front-left of target           |
| `up-right`                 | (1, -1, 0.3)         | Front-right of target          |


## Manual Environment Setup
For development or debugging, you can start components manually:

### Terminal 1 - Start UE4 Environment:
```bash
./scripts/start_bunker_env.sh /path/to/UnrealApp.sh
```

### Terminal 2 - Connect Drone Firmware:
```bash
# Default pose
./scripts/connect_drone_firmware.sh

# Specific starting position (See Test Scenarios table)
./scripts/connect_drone_firmware.sh --pose=left

# Custom pose (x y z roll pitch yaw)
./scripts/connect_drone_firmware.sh --custom "1.0 0.5 0.3 0 0 1.57"

# List available poses
./scripts/connect_drone_firmware.sh --list
```

### Terminal 3 - Run Controller:
```bash
python runnable/run_ibvs_splitter.py
# or
python runnable/run_student.py
```

See [scripts/README.md](../scripts/README.md) for additional details.

## Individual Controllers
For testing or debugging, run individual controllers directly:

### NSER-IBVS Splitter (Teacher)
```bash
python runnable/run_ibvs_splitter.py
```

This runs the full NSER-IBVS pipeline:
1. YOLO detects and segments the target vehicle
2. Mask Splitter divides segmentation into front/back regions
3. Sorted keypoints (that provide the car pose) are extracted from mask boundaries
4. IBVS control law computes velocity commands

### Distilled Student Neural Network
```
python runnable/run_student.py
```

This runs the lightweight student network (~1.7M parameters):
1. Takes raw camera frame as input
2. Directly outputs velocity commands
3. No explicit detection or IBVS computation


## Output Structure
Experiments generate outputs in the `output/` directory:
```bash
output/
└── {config-name}/
    ├── logs/
    │   └── {timestamp}/
    │       └── additional-flight-logs.log
    ├── parquet-logs/
    │   └── {timestamp}/
    │       └── logs.parquet          # Detection information, IBVS calculations, velocity per each frame
    └── results/
        └── {timestamp}/
            ├── frames-directory      # Flight raw frames stored ar .jpg or .png
            ├── flight_duration.json  # Time information about mission and status
            └── metadata.json         # Additional drone metadata provided by Parrot Olympe
```

## Evaluating Results TODO


## Visualizing Results TODO

## Real-World Experiments
For real hardware experiments, the same controllers work with physical drones:
```bash
# Connect to real Anafi drone (requires drone_base configuration)
python runnable/run_ibvs_splitter.py --is_real_world --experiment_name="real-ibvs-down-left"

python runnable/run_student.py --is_real_world --experiment_name="real-ibvs-down-left"
```

#### Experiment naming convention for real-world runs:
```bash
real-ibvs-{position}      # e.g., real-ibvs-down-left
real-student-{position}   # e.g., real-student-up-right
```

> **Note:** Pay attention before flying a real drone

## Configuration
Experiment behavior is controlled by YAML configuration files in [config/simulator/](../config/simulator):

See [Configuration Reference](configuration-reference.md) for details on all parameters.

## Common Issues

### Simulator Crashes on Startup
```bash
# Restart firmwared service
sudo systemctl restart firmwared.service
```

### Drone Not Responding
Check that:
1. Sphinx is running
2. Firmware URL is accessible (can also download it to avoid internet connection issues)
3. No other process is controlling the drone

### Detection Failures
If YOLO fails to detect the target:
- Verify the correct model file exists in `models/`
- Check camera stream is active (no streaming issues)
- Ensure proper lighting in the scene
- Ensure the testing scene is correct

### Student Network Produces Poor Commands
The student may need fine-tuning for your specific scenario. See
[student_train_pipeline/README.md](../student_train_pipeline/README.md) for training instructions.






