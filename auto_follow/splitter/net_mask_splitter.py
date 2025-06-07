from pathlib import Path

import torch
from torch import nn
from torchinfo import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with max pool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MaskSplitterNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 2,
            base_channels: int = 32,
            dropout_rate: float = 0.1,
            device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, base_channels, 0)
        self.image_conv = DoubleConv(3, base_channels, 0)
        self.mask_attention = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.down1 = Down(base_channels, base_channels * 2, 0)
        self.down2 = Down(base_channels * 2, base_channels * 4, 0)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout_rate)

        self.up1 = Up(base_channels * 8, base_channels * 4, dropout_rate)
        self.up2 = Up(base_channels * 4, base_channels * 2, 0)
        self.up3 = Up(base_channels * 2, base_channels, 0)

        self.outc = OutputConv(base_channels, out_channels)

        self.to(self.device)

    def forward(self, x):
        img, mask = x[:, :3], x[:, 3:]
        x_img = self.image_conv(img)
        attention = self.mask_attention(mask)
        x1 = x_img * (1 + attention)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

    def display_summary(self, input_shape: tuple[int, int, int, int]):
        summary(self, input_size=input_shape)

    def load_model(self, model_path: str | Path):
        self.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    input_channels = 4  # RGB (3) + mask (1)
    model = MaskSplitterNet(in_channels=input_channels, out_channels=2, base_channels=32)

    sample_input_size = (1, input_channels, 360, 640)
    sample_input = torch.randn(*sample_input_size)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    model.display_summary(sample_input_size)
