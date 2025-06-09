from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from auto_follow.splitter.net_mask_splitter import MaskSplitterNet


class MaskSplitterInference:
    def __init__(
            self,
            model_path: str | Path, device: torch.device | None = None,
            image_size: tuple[int, int] = (360, 640),
            confidence_threshold: float = 0.5,
            is_model_compiled: bool = False
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.is_model_compiled = is_model_compiled

        self.model = self._load_model()

    def preprocess(self, image: np.ndarray, mask: np.ndarray, is_resize: bool = True) -> torch.Tensor:
        """
        Prepares the input 4-channel tensor from image and mask.

        :param image: This uses an OpenCV based type of numpy ndarray.
        :param mask: Gray image binary mask as numpy ndarray. Represeting the segmentation of the car.
        :param is_resize: Whether the image is resized or not.
        :return: The prepared input for the MaskSplitterNet.
        """
        if is_resize:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))

        image = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).float() / 255.0
        return torch.cat([image, mask], dim=0).to(self.device).unsqueeze(0)

    def infer(self, image: np.ndarray, mask: np.ndarray, is_resize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Performs inference on a single frame."""
        input_tensor = self.preprocess(image, mask, is_resize)
        with torch.inference_mode():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output)
            predictions = (probs > self.confidence_threshold).to(dtype=torch.uint8).squeeze(0)
        return predictions[0].mul(255).cpu().numpy(), predictions[1].mul(255).cpu().numpy()

    def visualize(
            self,
            image_bgr: np.ndarray,
            front_mask: np.ndarray,
            back_mask: np.ndarray,
            figsize: tuple[int, int] = (10, 6)
    ):
        """Displays the input and predicted masks."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Front Prediction")
        plt.imshow(front_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Back Prediction")
        plt.imshow(back_mask, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _load_model(self) -> MaskSplitterNet:
        try:
            model = MaskSplitterNet(
                in_channels=4,
                out_channels=2,
                base_channels=32,
                dropout_rate=0.0,
                device=self.device
            )
            model.load_model(self.model_path)
            model.to(self.device)
            model.eval()
            if self.is_model_compiled:
                model = torch.compile(model)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def warm_up(self, image: np.ndarray, mask: np.ndarray, num_iterations: int = 10):
        """
        Warm up the model with dummy data for consistent timing.
        """
        print("Warming up model...")
        for _ in range(num_iterations):
            self.infer(image, mask)
        print("Warm-up complete!")
