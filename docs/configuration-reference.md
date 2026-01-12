# Configuration Reference
This document describes all configuration options available in the Auto-Follow framework.

## Configuration Files
Configuration files are organized in the [config/](../config) directory:
```bash
config/
├── pid/                    # PID controller parameters
│   ├── pid_x.yaml
│   ├── pid_x_lambda.yaml
│   ├── pid_y_lambda.yaml
│   ├── pid_forward.yaml
│   ├── low_pass_filter.yaml
│   └── low_pass_filter_lambda.yaml
└── simulator/              # Experiment scenario configurations
    ├── bunker-online-4k-config-default.yaml
    ├── bunker-online-4k-config-test-down-left.yaml
    ├── bunker-online-4k-config-test-down-right.yaml
    ├── bunker-online-4k-config-test-front-small-offset-left.yaml
    ├── bunker-online-4k-config-test-front-small-offset-right.yaml
    ├── bunker-online-4k-config-test-left.yaml
    ├── bunker-online-4k-config-test-right.yaml
    ├── bunker-online-4k-config-test-up-left.yaml
    ├── bunker-online-4k-config-test-up-right.yaml
    └── *-student.yaml      # Student variants of each scenario
```

> **Note:** Check [drone-sim-runner](https://github.com/SpaceTime-Vision-Robotics-Laboratory/drone-sim-runner) 
> repository for more details about the simulator `.yaml` files.

### Simulator Configuration
Simulator configs define experiment scenarios for the Parrot Sphinx simulator.

#### File Naming Convention
```
bunker-online-4k-config-{test-scenario}[-student].yaml
```

For the file name above each substring separated by `-` represents a part of the configuration:
- **bunker:** Uses the bunker custom-built Parrot Sphinx UE4 environment
- **online:** Uses the online version of the firmware (to use the offline one provide the downloaded firmware as path)
- **4k:** Uses the Anafi 4K drone firmware (can be Anafi AI, another Parrot supported firmware)
- **{test-scenario}:** Scenario name or starting position of the drone (See [running-experiments.md](running-experiments.md) for more information)
- **[-student]:** Optional, uses student network if existent, uses NSER-IBVS otherwise

## Reference Images
Reference images define the target view the drone should achieve.
Location: [assets/reference/](../assets/reference/)
```bash
reference/
├── images/
│   ├── frame_sim_45_5m_center.png      # Reference frame
│   └── ...
└── data/
    ├── frame_sim_45_5m_center.json     # Keypoint annotations
    └── ...
```

## Environment Configuration
Parrot Sphinx UE4 environment configs are in [assets/environment/config/](../assets/environment/config):
```bash
Meshes:
  - Name: "MeshName"
    FbxPath: "${MODELS_DIR}/mesh-name.fbx"
    Location: "0 0 12"
    Rotation: "0 0 -90"
    Scale: "1 1 1"
    SnapToGround: true
```

The `${MODELS_DIR}` placeholder is resolved at runtime to the absolute path of the 
[assets/environment/models/](../assets/environment/models) directory.

## Model Configuration
Model paths can be specified in simulator configs or as command-line arguments:

### YOLO Segmentation Models
| Model                                                   | Training Data | 
|---------------------------------------------------------|---------------|
| `29_05_best__yolo11n-seg_sim_car_bunker__all.pt`        | Simulator     |
| `real-yolo-car-full-segmentation.pt`                    | Real-world    |

### Mask Splitter Models

| Model                                                                       | Training Data |
|-----------------------------------------------------------------------------|---------------|
| `mask_splitter-epoch_10-dropout_0-low_x2-and-high_x0_quality_early_stop.pt` | Simulator     |
| `mask_splitter-epoch_10-dropout_0-_x2_real_early_stop.pt`                   | Real-world    |

### Student Models

| Model                                              | Training Data                                 |
|----------------------------------------------------|-----------------------------------------------|
| `student_model_sim_on_real_world_distribution.pth` | Simulator with real-world normalization       |
| `student_real_pretrained_augX3_80_runs.pth`        | Real-world fine-tuned from simulator pretarin |
| `sim_temporal_student_best_model.pth`              | Simulator with temporal context               |


## Creating Custom Configurations
### New Simulator Scenario
1. Copy an existing config
2. Modify drone pose for desired starting position
3. Update directory name

### Adding a New Reference Frame
1. Capture image using the `display_only_controller.py` from [external/drone_base/](../external/drone_base) 
`examples` directory. Follow that repository ready me for how to use keyboard to control the drone and saving images.
2. Run YOLO model to generate the segmentation and extract the keypoints saving them to JSON.
3. Add the new path to the JSON file to `IBVSYoloProcessor` parameter `goal_frame_points_path`.

