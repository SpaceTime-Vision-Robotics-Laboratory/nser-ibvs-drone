from pathlib import Path

import torch
from torch import nn
from torchinfo import summary


class DroneCommandRegressor(nn.Module):
    def __init__(self, img_size: tuple[int, int] = (224, 224)):
        super().__init__()
        self.img_size = img_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        infered_shape = self.conv_layers(torch.randn(1, 3, img_size[0], img_size[1])).shape

        self.fc_layers = nn.Sequential(
            nn.Linear(infered_shape[1] * infered_shape[2] * infered_shape[3], 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return torch.tanh(x)

    def display_summary(self):
        summary(self, input_size=(1, 3, self.img_size[0], self.img_size[1]))

    def load_model(self, model_path: str | Path):
        self.load_state_dict(torch.load(model_path)["model_state_dict"])


if __name__ == "__main__":
    model = DroneCommandRegressor()
    model.display_summary()
