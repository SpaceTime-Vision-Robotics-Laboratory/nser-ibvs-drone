import os
import unittest

import torch

from auto_follow.distiled_network.drone_command_temporal_regressor import TemporalDroneRegressor


class TestTemporalDroneRegressor(unittest.TestCase):
    def setUp(self):
        """Initialize the temporal model and dimensions."""
        self.img_size = (224, 224)
        self.batch_size = 2
        self.num_frames = 3
        self.prev_command_steps = 2
        self.model = TemporalDroneRegressor(img_size=self.img_size)

    def test_forward_output_shape(self):
        """Verify output shape (B, 3) for the fused temporal inputs."""
        images = torch.randn(self.batch_size, self.num_frames, 3, *self.img_size)
        prev_commands = torch.randn(self.batch_size, self.prev_command_steps, 3)

        output = self.model(images, prev_commands)

        expected_shape = (self.batch_size, 3)
        self.assertEqual(output.shape, expected_shape)

    def test_concatenation_logic(self):
        """Ensure the internal fusion logic produces the correct vector size before FC layers."""
        # Visual (256 * 3) + Commands (3 * 2) = 774
        expected_features = 774

        in_features = self.model.fc_layers[0].in_features
        self.assertEqual(in_features, expected_features,
                         f"Concatenation logic mismatch. Expected {expected_features}, got {in_features}")

    def test_temporal_sensitivity(self):
        """Verify that changing one frame in the sequence changes the output."""
        images = torch.randn(1, self.num_frames, 3, *self.img_size)
        prev_commands = torch.randn(1, self.prev_command_steps, 3)

        output_orig = self.model(images, prev_commands).detach()

        # Modify only the first frame (t-2)
        images_modified = images.clone()
        images_modified[0, 0] += 1.0

        output_mod = self.model(images_modified, prev_commands).detach()

        self.assertFalse(torch.allclose(output_orig, output_mod),
                         "Model output did not change when input sequence frame was modified.")

    def test_invalid_sequence_length(self):
        """Ensure the model raises an error if the sequence length is incorrect."""
        wrong_images = torch.randn(self.batch_size, 5, 3, *self.img_size)
        prev_commands = torch.randn(self.batch_size, self.prev_command_steps, 3)

        with self.assertRaises(ValueError):
            self.model(wrong_images, prev_commands)

    def test_save_load(self):
        """Test standard state_dict loading."""
        temp_path = "temporal_model.pth"
        torch.save(self.model.state_dict(), temp_path)

        try:
            self.model.load_model(temp_path)
        except Exception as e:
            self.fail(f"load_model failed: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
