import os
import unittest

import torch

from auto_follow.distiled_network.drone_command_regressor import DroneCommandRegressor


class TestDroneCommandRegressor(unittest.TestCase):
    def setUp(self):
        """Initialize model and common variables for tests."""
        self.img_size = (224, 224)
        self.model = DroneCommandRegressor(img_size=self.img_size)
        self.batch_size = 4

    def test_output_shape(self):
        """Verify the output shape is (Batch, 3)."""
        x = torch.randn(self.batch_size, 3, *self.img_size)
        output = self.model(x)

        expected_shape = (self.batch_size, 3)
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape}, got {output.shape}")

    def test_output_range(self):
        """Verify output is within [-1, 1] due to tanh activation."""
        x = torch.randn(self.batch_size, 3, *self.img_size)
        output = self.model(x)

        self.assertTrue(torch.all(output >= -1.0) and torch.all(output <= 1.0),
                        "Output values outside the range of [-1, 1]")

    def test_different_input_sizes(self):
        """Test if the model handles a different image size correctly (e.g., 128x128)."""
        custom_size = (128, 128)
        model = DroneCommandRegressor(img_size=custom_size)
        x = torch.randn(1, 3, *custom_size)

        try:
            output = model(x)
            self.assertEqual(output.shape, (1, 3))
        except Exception as e:
            self.fail(f"Model failed with custom image size {custom_size}: {e}")

    def test_parameter_update(self):
        """Check if weights actually update during a mock training step."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        x = torch.randn(self.batch_size, 3, *self.img_size)
        target = torch.randn(self.batch_size, 3)

        initial_params = self.model.conv_layers[0].weight.clone()

        optimizer.zero_grad()
        output = self.model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        updated_params = self.model.conv_layers[0].weight

        self.assertFalse(torch.equal(initial_params, updated_params),
                         "Model weights did not update after backward pass.")

    def test_save_load_logic(self):
        """Test the load_model method with a dummy state dict."""
        temp_path = "temp_model.pth"

        dummy_state = {"model_state_dict": self.model.state_dict()}
        torch.save(dummy_state, temp_path)

        try:
            self.model.load_model(temp_path)
        except Exception as e:
            self.fail(f"load_model failed: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
