from pathlib import Path

import torch
from torch import nn
from torchinfo import summary


class TemporalDroneRegressor(nn.Module):
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

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        visual_feature_size = 256 * 3
        # We use 2 previous commands, each with 3 values (x,y,rot)
        command_feature_size = 3 * 2

        total_input_features = visual_feature_size + command_feature_size  # 768 + 6 = 774

        self.fc_layers = nn.Sequential(
            nn.Linear(total_input_features, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, images: torch.Tensor, prev_commands: torch.Tensor) -> torch.Tensor:
        """
        :param images: Tensor of shape (B, 3, C, H, W) for the 3 frames.
        :param prev_commands: Tensor of shape (B, 2, 3) for the 2 previous commands.
        """
        # We expect images to be (B, Seq=3, C, H, W)
        # We process each of the 3 images in the sequence through the CNN
        batch_size = images.shape[0]

        # Unbind the sequence of images and process each one
        img_t2, img_t1, img_t0 = torch.unbind(images, dim=1)

        features_t2 = self.conv_layers(img_t2).view(batch_size, -1)  # (B, 256)
        features_t1 = self.conv_layers(img_t1).view(batch_size, -1)  # (B, 256)
        features_t0 = self.conv_layers(img_t0).view(batch_size, -1)  # (B, 256)

        # Concatenate the visual features from all three frames
        visual_features = torch.cat([features_t2, features_t1, features_t0], dim=1)

        # Flatten the previous commands tensor from (B, 2, 3) to (B, 6)
        command_features = prev_commands.view(batch_size, -1)

        combined_features = torch.cat([visual_features, command_features], dim=1)

        output = self.fc_layers(combined_features)
        return torch.tanh(output)

    def display_summary(self):
        dummy_images = torch.zeros((1, 3, 3, self.img_size[0], self.img_size[1]))  # (B=1, Seq=3, C=3, H, W)
        dummy_commands = torch.zeros((1, 2, 3))  # (B=1, Seq=2, Values=3)

        summary(self, input_data=[dummy_images, dummy_commands])

    def load_model(self, model_path: str | Path):
        self.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    model = TemporalDroneRegressor()
    model.display_summary()
