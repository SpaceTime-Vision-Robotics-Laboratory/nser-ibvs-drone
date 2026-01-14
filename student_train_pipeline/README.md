# Student Distil Training Pipeline

This pipeline trains the **lightweight** student network (~1.7M parameters) to mimic 
the **NSER-IBVS** teacher velocity commands through knowledge distillation.

![Student Network Architecture](../docs/images/Student-Architecture.png)


## Overview
The student learns to directly regress drone velocity commands (vx, vy, ωz) from 
RGB images without explicit visual servoing computation, achieving **11x faster 
inference** than the teacher pipeline.

## Dataset Preparation

### Step 1: Collect Training Data
Run NSER-IBVS experiments to generate training data:
```bash
python runnable/run_simulator_experiments.py \
    --sphinx_bunker_base_dir=/path/to/UnrealApp.sh \
    --target_runs=30
```

Needed files that this generates:
- Raw frames in `output/{config}/results/{timestamp}/frames/`
- Velocity labels in `output/{config}/parquet-logs/{timestamp}/logs.parquet`

### Step 2: Transform to Training Format
```bash
python dataset_prepare.py \
    --raw_data_path=/path/to/raw_experiment_output \
    --output_path=/path/to/training_dataset
```

**Output Structure:**
```
output_path/
├── train/
│   └── {scene_name}/
│       └── RUN_{timestamp}/
│           ├── images/
│           │   └── image_frame_XXXXXX.jpg
│           └── labels/
│               └── image_frame_XXXXXX.txt  # Contains: x_cmd y_cmd rot_cmd
└── validation/
    └── ... (same structure, last 2 runs per scene)
```

### Step 3: Validate Dataset
To check if the dataset is prepared correctly, you can use
```bash
python drone_dataset.py
```

## Training

### Training Configuration

| Parameter      | Value               | Description                                           |
|----------------|---------------------|-------------------------------------------------------|
| Optimizer      | Adam                | -                                                     |
| Learning Rate  | 0.001               | -                                                     |
| Batch Size     | 128                 | -                                                     |
| Loss Function  | MSE                 | Mean Squared Error                                    |
| Early Stopping | patience=3, δ=10^-4 | -                                                     |
| Input Size     | 224×224             | RGB images                                            |
| Normalization  | ImageNet            | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### Velocity Command Normalization
Commands are normalized to [-1, 1] range for training stability and to match the 
actual drone command bounds (-100, 100):

| Command | Normalization Factor (Sim \| Real) | Range            |
|---------|------------------------------------|------------------|
| vx      | +/- 24.0 \| 30                     | Forward/backward |
| vy      | +/- 9.0  \| 30                     | Left/right       |
| ωz      | +/- 40.0 \| 40                     | Yaw rotation     |

> **Note:** These factors were empirically determined from the distribution of 
> NSER-IBVS outputs across all simulation experiments.

## Run Training
```bash
python train_model.py
```

**Outputs:**
- Checkpoints: `runs/run_{timestamp}/checkpoints/`
- TensorBoard logs: `runs/run_{timestamp}/tensorboard/`
- Training logs: `runs/run_{timestamp}/training.log`

### Monitor Training
```bash
tensorboard --logdir=runs/
```

## Model Architecture
The `DroneCommandRegressor` is a lightweight CNN:
```
Input (3, 224, 224)
    ↓
Conv2d(3→16, k=5) → BN → GELU → MaxPool
Conv2d(16→32, k=5) → BN → GELU → MaxPool
Conv2d(32→64, k=3) → BN → GELU
Conv2d(64→128, k=3) → BN → GELU
Conv2d(128→256, k=3) → BN → GELU
Conv2d(256→512, k=3) → BN → GELU
    ↓
AdaptiveAvgPool2d(1, 1)
    ↓
Linear(512→256) → GELU
Linear(256→3) → Tanh
    ↓
Output (vx, vy, ωz) ∈ [-1, 1]
```

**Total Parameters:** ~1.7M

## Inference
After training, the model outputs normalized commands. De-normalize before sending to drone:
```python
vx_actual = output[0] * -24.0
vy_actual = output[1] * 9.0
omega_z_actual = output[2] * 41.0
```

## Two-Stage Training (Sim-to-Real)
For real-world deployment:
1. **Stage 1:** Pretrain on simulation data (240 runs, ~74K frames)
2. **Stage 2:** Fine-tune on real-world data (80 runs, ~44K frames)

The real-world fine-tuning uses the same architecture but may use different 
normalization bounds (see paper Table 1 for real-world ranges).
