import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from torchvision import transforms


class DroneCommandDataset(Dataset):
    """
    PyTorch Dataset for Drone Command Dataset.
    
    Dataset structure:
    - Each scene contains multiple runs
    - Each run has images/ and labels/ directories
    - Images are JPG files named image_frame_XXXXXX.jpg
    - Labels are text files with x, y, rot values
    """
    
    def __init__(
        self,
        data_root: str,
        scenes: list[str] | None = None,
        transform: transforms.Compose | None = None,
    ):
        """
        Initialize the DroneCommandDataset.
        
        Args:
            data_root (str): Root directory containing scene folders
            scenes (List[str], optional): List of scene names to include. If None, includes all scenes
        """
        self.data_root = Path(data_root)
        self.runs = []
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Get all scene directories if not specified
        if scenes is None:
            scenes = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        
        self._collect_samples(scenes)
        print(f"Collected {len(self.samples)} samples from {len(scenes)} scenes and {len(self.runs)} runs")
        
    def _collect_samples(self, scenes: list[str]) -> list[tuple[str, torch.Tensor]]:
        for scene in scenes:
            scene_path = self.data_root / scene
            for run in scene_path.iterdir():
                if run.is_dir():
                    self.runs.append(run)
                    for label_path, image_path in zip(run.glob("labels/*.txt"), run.glob("images/*.jpg")):
                        if label_path.stem != image_path.stem:
                            print(f"Label file {label_path} does not match image file {image_path} for scene {scene} run {run}")
                            continue
                        self.samples.append((image_path, self._load_label(label_path)))
        

    def _load_label(self, label_path: str) -> torch.Tensor:
        """Load and parse label from text file. Keep as integers."""
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    values = [int(float(x)) for x in line.split()]  # Convert to int, handling float format
                    if len(values) >= 3:
                        label = np.array(values[:3], dtype=np.float32)  # x, y, rot as integers
                        return torch.from_numpy(label)
                    else:
                        raise ValueError(f"Expected at least 3 values, got {len(values)}")
                else:
                    raise ValueError("Empty label file")
        except Exception as e:
            print(f"Error loading label from {label_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(3, dtype=torch.int32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        Normalize the label to range [-1, 1] by dividing by the max value of the dataset.
        
        Returns:
            Dictionary containing:
                - 'image': Transformed image tensor
                - 'label': Label tensor [x, y, rot] as integers
        """
        image_path, label = self.samples[idx]
        
        label[0] = label[0] / -24.0
        label[1] = label[1] / 9.0
        label[2] = label[2] / 41.0
        
        image = cv2.imread(image_path)
        image = image.astype(np.float32) / 255.0    
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        image_tensor = self.transform(image_tensor)
        
        return image_tensor, label

def validate_dataset(dataset: DroneCommandDataset) -> list[int]:
    from tqdm import tqdm
    
    errors = []
    for idx in tqdm(range(len(dataset)), desc="Validating dataset"):
        try:
            dataset[idx]
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            errors.append(idx)
    return errors

def show_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    import matplotlib.pyplot as plt
    
    images, labels = batch
    batch_size = images.shape[0]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    for idx in range(batch_size):
        row, col = idx // grid_size, idx % grid_size
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Label: {labels[idx].tolist()}")
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for idx in range(batch_size, grid_size * grid_size):
        row, col = idx // grid_size, idx % grid_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example of how to use the dataset and dataloader
    
    data_root = "sim_ds/train"
    dataset = DroneCommandDataset(data_root=data_root)
    
    errors = validate_dataset(dataset)
    print(f"Found {len(errors)} errors in the dataset, {len(dataset) - len(errors)} samples are valid")
    print(f"Collected {len(dataset)} samples from {len(dataset.runs)} runs")
    
    data_root = "sim_ds/validation"
    dataset = DroneCommandDataset(data_root=data_root)
    print(f"Collected {len(dataset)} samples from {len(dataset.runs)} runs")
    
    errors = validate_dataset(dataset)