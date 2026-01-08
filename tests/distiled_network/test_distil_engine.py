import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from auto_follow.distiled_network.distil_engine import StudentEngine


@patch("auto_follow.distiled_network.distil_engine.DroneCommandRegressor")
@patch("torch.load")
class TestStudentEngine(unittest.TestCase):

    def setUp(self):
        """Set up the engine once for standard tests."""
        self.image_size = (224, 224)

        with patch("torch.load") as mock_load, \
                patch("auto_follow.distiled_network.distil_engine.DroneCommandRegressor") as mock_reg:
            mock_load.return_value = {"model_state_dict": {}}
            self.mock_model = MagicMock()
            mock_reg.return_value = self.mock_model

            self.engine = StudentEngine(model_path="dummy.pth", image_size=self.image_size)

    def test_preprocess_output_format(self, mock_load, mock_reg):
        """Verify preprocess converts a NumPy image to a normalized FloatTensor."""
        dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        processed = self.engine.preprocess(dummy_img)

        self.assertEqual(processed.shape, (1, 3, 224, 224))
        self.assertEqual(processed.device.type, self.engine.device.type)

    def test_predict_scaling(self, mock_load, mock_reg):
        """Verify the output is scaled by the denormalize factors."""
        fixed_output = torch.tensor([[1.0, -1.0, 0.5]], device=self.engine.device)
        self.mock_model.return_value = fixed_output

        dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        prediction = self.engine.predict(dummy_img)

        # Expected: [1.0*30, -1.0*30, 0.5*40] -> [30, -30, 20]
        expected_output = np.array([30.0, -30.0, 20.0])
        np.testing.assert_allclose(prediction, expected_output, atol=1e-5)

    def test_device_management(self, mock_load, mock_reg):
        """Check if engine correctly assigns the requested device."""
        mock_load.return_value = {"model_state_dict": {}}
        engine_cpu = StudentEngine(model_path="dummy.pth", device="cpu")
        self.assertEqual(engine_cpu.device.type, "cpu")

    def test_inference_mode_active(self, mock_load, mock_reg):
        """Ensure model is in eval mode."""
        self.mock_model.eval.assert_called()


if __name__ == "__main__":
    unittest.main()