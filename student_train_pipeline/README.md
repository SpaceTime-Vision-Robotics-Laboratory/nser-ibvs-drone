# Student Distil Training Pipeline

## How to run
Prepare the dataset by running the `dataset_prepare.py` script.

```bash
python dataset_prepare.py --raw_dataset_path /path/to/raw_dataset --output_path /path/to/output
```

## Validate the dataset

To check if the dataset is prepared correctly, you can use the `drone_dataset.py` script.

```bash
python drone_dataset.py
```

## How to run the training
Set the dataset path in the `train_model.py` script.

```bash
python train_model.py
```

```python
dataset_train = DroneCommandDataset(data_root="sim_ds/train")
```

```python
dataset_val = DroneCommandDataset(data_root="sim_ds/validation")
```