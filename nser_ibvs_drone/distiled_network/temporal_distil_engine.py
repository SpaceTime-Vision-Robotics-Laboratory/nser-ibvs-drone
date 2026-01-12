from collections import deque
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from nser_ibvs_drone.distiled_network.drone_command_temporal_regressor import TemporalDroneRegressor


class TemporalStudentEngine:
    def __init__(self, model_path: str | Path, device: str | None = None, image_size: tuple[int, int] = (224, 224)):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model_path = model_path
        self.model = self._load_model()
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.denormalize = torch.tensor([9.0, 5.0, 24.0], device=self.device)

        self.image_buffer = deque(maxlen=3)
        self.command_buffer = deque(maxlen=2)
        zero_image = torch.zeros((3, *image_size), device=self.device)
        zero_command = torch.zeros(3, device=self.device)
        for _ in range(3):
            self.image_buffer.append(zero_image)
        for _ in range(2):
            self.command_buffer.append(zero_command)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0
        image = self.transform(image)
        self.image_buffer.append(image)
        return image

    def predict(self, image: np.ndarray) -> torch.Tensor:
        self.preprocess(image)

        images = torch.stack(list(self.image_buffer), dim=0).unsqueeze(0) # (1, 3, C, H, W)
        commands = torch.stack(list(self.command_buffer), dim=0).unsqueeze(0) # (1, 2, 3)

        with torch.inference_mode():
            output = self.model(images, commands)
            print(f"{output=}")
            output *= self.denormalize
            output = output.squeeze(0).cpu().numpy()
        self.command_buffer.append(torch.from_numpy(output).to(self.device))
        return output

    def _load_model(self) -> TemporalDroneRegressor:
        model = TemporalDroneRegressor()
        model.load_model(self.model_path)
        model.eval()
        model.to(self.device)
        return model
