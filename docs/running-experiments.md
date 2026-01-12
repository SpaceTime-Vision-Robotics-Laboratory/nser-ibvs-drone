# Running Experiments
This guide explains how to run visual servoing experiments using `NSER-IBVS-Drone` framework.

<table align="center">
  <tr>
    <td align="center"><img src="gifs/Real-IBVS-Front-Left.gif" width="200" alt="NSER-IBVS Front Left"><br>NSER-IBVS Front Left</td>
    <td align="center"><img src="gifs/Real-Student-Front-Left.gif" width="200" alt="Student Front Left"><br>Student Front Left</td>
    <td align="center"><img src="gifs/Real-IBVS-Front-Right.gif" width="200" alt="NSER-IBVS Front Right"><br>NSER-IBVS Front Right</td>
    <td align="center"><img src="gifs/Real-Student-Front-Right.gif" width="200" alt="Student Front Right"><br>Student Front Right</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center"><img src="gifs/Real-IBVS-Up-Left.gif" width="200" alt="NSER-IBVS Up Left"><br>NSER-IBVS Up Left</td>
    <td align="center"><img src="gifs/Real-Student-Up-Left.gif" width="200" alt="Student Up Left"><br>Student Up Left</td>
    <td align="center"><img src="gifs/Real-IBVS-Up-Right.gif" width="200" alt="NSER-IBVS Up Right"><br>NSER-IBVS Up Right</td>
    <td align="center"><img src="gifs/Real-Student-Up-Right.gif" width="200" alt="Student Up Right"><br>Student Up Right</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center"><img src="gifs/Real-IBVS-Front-Center.gif" width="400" alt="NSER-IBVS Front Center (Fail)"><br>NSER-IBVS Front Center (Fail)</td>
    <td align="center"><img src="gifs/Real-Student-Front-Center.gif" width="400" alt="Student Front Center"><br>Student Front Center</td>
  </tr>
</table>



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
- Downloaded the [bunker custom-built UE4 environment](https://drive.google.com/file/d/1kHqJtTq7CGoazUUn8tPFnijV3lYY4toO/view?usp=drive_link)
- Enough disk space if desired to run automated experiments (~150GB per 100 runs)

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

> **Note:** This will generate quite a few logs and frames ~150GB per 100 runs. Make sure you have enough space.

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

## Evaluating Results
The framework provides two types of evaluation: **Statistics Evaluation** for analyzing flight 
performance metrics, and **Inference Evaluation** for benchmarking model computational efficiency.

### Statistics Evaluation
The statistics evaluation analyzes experiment results to compute performance metrics including 
visual servoing errors, IoU scores, command statistics, flight duration, and distance traveled.

#### Running Statistics Evaluation:
```bash
python runnable/run_statistics_evaluation.py \
    --results_dir=/path/to/experiment-results \
    --scene_name=sim-student-results \
    --save_dir=./results/statistics \  # Optional, uses results/statistics by default.
    --is_student \   # Optional, if given will use student configuration else IBVS teacher configuration
    --is_real_world  # Optional, if given will use real-world configuration else digital-twin configuration
```

**Arguments:**

| Argument          | Description                                                  | Default              |
|-------------------|--------------------------------------------------------------|----------------------|
| `--results_dir`   | Path to directory containing experiment results              | Required             |
| `--scene_name`    | Name for the output directory (see scene names below)        | Required             |
| `--save_dir`      | Directory to save evaluation results                         | `results/statistics` |
| `--is_student`    | Use Student model configuration (omit for NSER-IBVS Teacher) | False                |
| `--is_real_world` | Use Real-World configuration (omit for Simulator)            | False                |


**Scene names convention:**
- `sim-ibvs-results` - Simulator NSER-IBVS Teacher
- `sim-student-results` - Simulator Student
- `real-ibvs-results` - Real-World NSER-IBVS Teacher
- `real-student-results` - Real-World Student

#### Output Metrics

The statistics evaluation generates the following CSV files and plots:

| Output File                        | Description                                                 |
|------------------------------------|-------------------------------------------------------------|
| `*_error_stats-total.csv`          | Per-run error norm statistics (mean, median, std, min, max) |
| `*_error_stats-mean.csv`           | Mean error statistics grouped by direction                  |
| `*_iou_stats-total.csv`            | Per-run IoU statistics for last 3 seconds                   |
| `*_iou_stats-mean.csv`             | Mean IoU statistics grouped by direction                    |
| `*_cmd_stats-total.csv`            | Command statistics (x, y, rotation commands)                |
| `*_cmd_stats-mean.csv`             | Mean command statistics per run                             |
| `*_duration_stats-total.csv`       | Flight duration per run                                     |
| `*_duration_stats-mean_median.csv` | Duration statistics grouped by direction                    |
| `*_distance_stats-total.csv`       | Distance traveled per run                                   |
| `*_distance_stats-mean_median.csv` | Distance statistics grouped by direction                    |

**Generated Plots:**
- `last_3_seconds_iou_mean_per_direction.png` - Mean IoU bar chart by starting direction
- `last_3_seconds_iou_median_per_direction.png` - Median IoU bar chart by starting direction
- `last_3_seconds_err_uv_mean_per_direction.png` - Mean error norm by starting direction
- `last_3_seconds_err_uv_median_per_direction.png` - Median error norm by starting direction

### Inference Evaluation
The inference evaluation benchmarks computational performance of the different pipelines: 
Student, Student+Segmentation, and NSER-IBVS.

#### Running Inference Evaluation

```bash
python runnable/run_inference_evaluation.py \
    --frames_dir=/path/to/frames \
    --target_runs=30 \
    --save_dir=./results/inference
```

**Arguments:**

| Argument        | Description                                          | Default              |
|-----------------|------------------------------------------------------|----------------------|
| `--frames_dir`  | Path to directory containing test frames (.jpg/.png) | Required             |
| `--target_runs` | Number of evaluation trials per benchmark            | 30                   |
| `--save_dir`    | Directory to save evaluation results                 | `results/statistics` |

#### Benchmarks Performed

The script runs four evaluation types:
1. **Multi-Trial Inference Timing** ([eval_inference_multi_trials.py](../nser_ibvs_drone/evaluation/eval_inference_multi_trials.py))
   - Measures inference time per frame across multiple trials
   - Outputs: `benchmark_results.csv`, `benchmark_table.tex`

2. **Detailed Multi-Trial Benchmark** ([eval_inference_multi_trials_detailed.py](../nser_ibvs_drone/evaluation/eval_inference_multi_trials_detailed.py))
   - Comprehensive timing with per-run breakdowns
   - Outputs: `benchmark_detailed_*.json`, `benchmark_summary_*.csv`, `benchmark_per_run_*.csv`

3. **FLOPs and Parameters** ([evaluate_flops.py](../nser_ibvs_drone/evaluation/evaluate_flops.py))
   - Computes computational complexity (GFLOPs, TFLOPs) and model parameters
   - Outputs: `benchmark_timing_raw.csv`, `benchmark_complexity_summary.csv`, `benchmark_complexity_table.tex`

4. **Memory Usage** ([evaluation_memory.py](../nser_ibvs_drone/evaluation/evaluation_memory.py))
   - Measures RAM and GPU memory consumption
   - Outputs: `benchmark_memory.csv`

### Running Individual Benchmarks

You can also run individual evaluation scripts directly :

```bash
python -c "from nser_ibvs_drone.evaluation.eval_inference_multi_trials import benchmark_infer_multi_evaluators; benchmark_infer_multi_evaluators('/path/to/frames', trials=30, output_dir='./results/inference)"
```

#### Output Metrics

| Metric          | Description                                           |
|-----------------|-------------------------------------------------------|
| Average (ms)    | Mean inference time per frame                         |
| Median (ms)     | Median inference time per frame                       |
| Std (ms)        | Standard deviation of inference times                 |
| Min/Max (ms)    | Minimum and maximum inference times                   |
| FPS             | Frames per second (1000 / average_ms)                 |
| GFLOPs          | Billions of floating-point operations                 |
| Params (M)      | Model parameters in millions                          |
| RAM (MB)        | System memory usage                                   |
| GPU (MB)        | GPU memory usage (if available)                       |


## Visualizing Results

The framework provides multiple visualization tools for analyzing experiment results: 
distribution analysis for individual methods, teacher-student comparisons, and trajectory 
visualizations.

### Method Results Distributions Analysis
Generates comprehensive plots analyzing flight performance for a single method (Teacher or Student).

```bash
python runnable/visualization/run_method_distributions_analysis.py \
    --base_experiments_dir=/path/to/experiments \
    --save_dir=./results/plots \
    --is_real_world \
    --is_student \
    --random_runs=5
```

**Generated Plots:**

| Plot File                             | Description                                       |
|---------------------------------------|---------------------------------------------------|
| `flight_durations.png`                | Bar chart of flight duration by direction         |
| `flight_durations_by_direction.png`   | Box plot of duration distribution by direction    |
| `flight_durations_histogram.png`      | Histogram of overall flight duration distribution |
| `distances_by_direction.png`          | Bar chart of distance traveled by direction       |
| `distances_by_direction_boxplot.png`  | Box plot of distance distribution by direction    |
| `distances_histogram.png`             | Histogram of overall distance distribution        |
| `command_distributions.png`           | Histograms of x_cmd, y_cmd, rot_cmd distributions |
| `error_norm_runs.png`                 | Histogram of tracking error norms across all runs |
| `evolution_*_rl_style.png`            | Individual command/error evolution plots          |
| `evolution_*_commands_*_combined.png` | Combined 2x2 evolution plot                       |
| `time_series_sample_run_*.png`        | Time series plots for random individual runs      |

> **Note:** If analyzing Student experiments, ensure Teacher logs have been generated using
> [nser_ibvs_drone/evaluation/ibvs_splitter_run_for_logs.py](../nser_ibvs_drone/evaluation/ibvs_splitter_run_for_logs.py) 
> for proper comparison data.

### Teacher-Student Comparison
Generates side-by-side and combined comparison plots between Teacher (NSER-IBVS) and Student methods.

```bash
python runnable/visualization/run_teacher_student_comparison.py \
    --teacher_exp_path=/path/to/teacher/experiments \
    --student_exp_path=/path/to/student/experiments \
    --save_dir=./results/plots \
    --is_real_world
```

**Generated Comparison Plots:**

| Plot File                                        | Description                                      |
|--------------------------------------------------|--------------------------------------------------|
| `flight_durations_comparison.png`                | Bar chart comparing duration by direction        |
| `flight_durations_histogram_comparison.png`      | KDE overlay of duration distributions            |
| `distances_comparison.png`                       | Bar chart comparing distance by direction        |
| `distances_histogram_comparison.png`             | KDE overlay of distance distributions            |
| `error_norms_comparison.png`                     | Bar chart comparing error norms (last 3 seconds) |
| `error_norms_histogram_comparison.png`           | KDE overlay of error norm distributions          |
| `iou_comparison.png`                             | Bar chart comparing IoU values by direction      |
| `iou_histogram_comparison.png`                   | KDE overlay of IoU distributions                 |
| `command_distributions_comparison.png`           | KDE comparison of command distributions          |
| `command_distributions_magnitude_comparison.png` | Histogram comparison of command magnitudes       |


### Trajectory Visualization

#### Single Run Trajectory

Visualize the trajectory from a single experiment run.

```bash
python runnable/visualization/run_single_trajectory_analysis.py \
    --metadata_path=/path/to/experiment/results/timestamp/metadata.json \
    --direction=front-left \
    --goal=2.0,2.5 \
    --save_path=./results/trajectories/single-trajectory.png
```


**Arguments:**

| Argument          | Description                                                 | Default                                                 |
|-------------------|-------------------------------------------------------------|---------------------------------------------------------|
| `--metadata_path` | Path to `metadata.json` containing drone telemetry logs     | Required                                                |
| `--direction`     | Starting direction (predefined name or custom `x,y` offset) | Required                                                |
| `--goal`          | Goal position as `x,y` coordinates                          | `2.0,2.5`                                               |
| `--save_path`     | Path to save the output plot                                | `results/trajectories/individual/single-trajectory.png` |

**Available Predefined Directions:**
- `front-right`, `front-left`
- `left`, `right`
- `up-left`, `up-right`
- `down-left`, `down-right`

**Custom Direction:** You can specify a custom offset as `x,y` (e.g., `1.5,0.3` for a custom starting position relative to goal).

**Generated Plot:**

The single trajectory plot includes 4 subplots:
1. Altitude vs Time
2. Speed vs Time
3. 2D Trajectory (carpet coordinates or GPS)
4. Flight States over Time

#### Teacher-Student Trajectory Comparison
Generate trajectory comparison plots between Teacher and Student methods across all direction pairs.

```bash
python runnable/visualization/run_teacher_student_trajectory_analysis.py \
    --teacher_exp_path=/path/to/teacher/experiments \
    --student_exp_path=/path/to/student/experiments \
    --save_dir=./results/trajectories
```

**Arguments:**

| Argument             | Description                                          | Default                |
|----------------------|------------------------------------------------------|------------------------|
| `--teacher_exp_path` | Path to Teacher NSER-IBVS experiments base directory | Required               |
| `--student_exp_path` | Path to Student experiments base directory           | Required               |
| `--save_dir`         | Directory to save generated trajectory plots         | `results/trajectories` |

**Generated Trajectory Comparison Plots:**

| Plot File                                                | Description                             |
|----------------------------------------------------------|-----------------------------------------|
| `trajectories-comparison-teacher-student-front.png`      | Front-left and front-right trajectories |
| `trajectories-comparison-teacher-student-up.png`         | Up-left and up-right trajectories       |
| `trajectories-comparison-teacher-student-left-right.png` | Left and right trajectories             |
| `trajectories-comparison-teacher-student-down.png`       | Down-left and down-right trajectories   |

Each comparison plot shows:
- Mean trajectories with confidence bands
- Start points (green markers)
- Goal position (magenta star)
- Car position (with optional image overlay)
- Carpet boundary (dashed rectangle for simulator)


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






