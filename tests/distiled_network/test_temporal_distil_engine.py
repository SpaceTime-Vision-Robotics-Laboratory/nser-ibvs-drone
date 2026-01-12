import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nser_ibvs_drone.distiled_network.temporal_distil_engine import TemporalStudentEngine


@patch("nser_ibvs_drone.distiled_network.temporal_distil_engine.TemporalDroneRegressor")
@patch("torch.load")
class TestTemporalStudentEngine(unittest.TestCase):

    def setUp(self):
        """Initialize dimensions and paths."""
        self.image_size = (224, 224)
        self.model_path = "dummy_temporal.pth"

        with patch("torch.load") as mock_load, \
                patch("nser_ibvs_drone.distiled_network.temporal_distil_engine.TemporalDroneRegressor") as mock_reg:
            mock_load.return_value = {"model_state_dict": {}}
            self.mock_model = MagicMock()
            mock_reg.return_value = self.mock_model
            self.engine = TemporalStudentEngine(self.model_path, image_size=self.image_size)

    def test_initialization_buffers(self, mock_load, mock_reg):
        """Verify that buffers are initialized with zeros and correct lengths."""
        self.assertEqual(len(self.engine.image_buffer), 3)
        self.assertEqual(len(self.engine.command_buffer), 2)

        for img in self.engine.image_buffer:
            self.assertTrue(torch.all(img == 0))
            self.assertEqual(img.device.type, self.engine.device.type)

    def test_buffer_sliding_window(self, mock_load, mock_reg):
        """Verify that new images are appended and old ones are dropped (FIFO)."""
        ones_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        self.engine.preprocess(ones_img)

        self.assertFalse(torch.all(self.engine.image_buffer[-1] == 0))
        self.assertTrue(torch.all(self.engine.image_buffer[0] == 0))

    def test_predict_flow_and_command_feedback(self, mock_load, mock_reg):
        """Verify that model output is added back to the command buffer."""
        # In predict(), this is scaled by [9.0, 5.0, 24.0] -> [4.5, 2.5, 12.0]
        mock_output = torch.tensor([[0.5, 0.5, 0.5]], device=self.engine.device)
        self.mock_model.return_value = mock_output

        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        prediction = self.engine.predict(dummy_img)
        expected_scaled = np.array([4.5, 2.5, 12.0])
        np.testing.assert_allclose(prediction, expected_scaled)

        last_command = self.engine.command_buffer[-1]
        np.testing.assert_allclose(last_command.cpu().numpy(), expected_scaled)

    def test_predict_input_shapes(self, mock_load, mock_reg):
        """Ensure the model receives (1, 3, C, H, W) and (1, 2, 3) tensors."""
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        self.mock_model.return_value = torch.zeros((1, 3), device=self.engine.device)
        self.engine.predict(dummy_img)

        args, kwargs = self.mock_model.call_args
        images_tensor, commands_tensor = args

        self.assertEqual(images_tensor.shape, (1, 3, 3, 224, 224))
        self.assertEqual(commands_tensor.shape, (1, 2, 3))

    def test_device_consistency(self, mock_load, mock_reg):
        """Verify all internal tensors stay on the assigned device."""
        self.assertEqual(self.engine.denormalize.device.type, self.engine.device.type)
        for cmd in self.engine.command_buffer:
            self.assertEqual(cmd.device.type, self.engine.device.type)


if __name__ == "__main__":
    unittest.main()
