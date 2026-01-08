import importlib.util
import sys
import unittest


class TestBasicImport(unittest.TestCase):
    def test_basic_import(self):
        """Test that the module can be imported without errors."""
        try:
            from drone_base._version import __version__  # noqa: PLC0415
            self.assertIsNotNone(__version__, "Version should not be None")
            self.assertTrue(isinstance(__version__, str), "Version should be a string")
        except ImportError as e:
            self.fail(f"Import failed with error: {e}")

    def test_drone_base_module_existence(self):
        """Verify the module exists in sys.modules after import."""
        try:
            import drone_base  # noqa: F401, PLC0415
            self.assertIn('drone_base', sys.modules, "Module should be in sys.modules")
        except ImportError as e:
            self.fail(f"Import failed with error: {e}")

    def test_mask_splitter_module_existence(self):
        """Verify the module exists in sys.modules after import."""
        try:
            import mask_splitter  # noqa: F401, PLC0415
            self.assertIn('mask_splitter', sys.modules, "Module should be in sys.modules")
        except ImportError as e:
            self.fail(f"Import failed with error: {e}")

    def test_drone_sim_runner_module_existence(self):
        """Verify the module exists in sys.modules after import."""
        try:
            import drone_sim_runner  # noqa: F401, PLC0415
            self.assertIn('drone_sim_runner', sys.modules, "Module should be in sys.modules")
        except ImportError as e:
            self.fail(f"Import failed with error: {e}")

    def test_importlib_drone_base_approach(self):
        """Test import using importlib for more detailed error reporting."""
        self._importlib_approach_helper("drone_base._version")

    def test_importlib_mask_splitter_approach(self):
        """Test import using importlib for more detailed error reporting."""
        self._importlib_approach_helper("mask_splitter._version")

    def test_importlib_drone_sim_runner_approach(self):
        """Test import using importlib for more detailed error reporting."""
        self._importlib_approach_helper("drone_sim_runner._version")

    def _importlib_approach_helper(self, module_name: str):
        try:
            spec = importlib.util.find_spec(module_name)
            self.assertIsNotNone(spec, f"Module spec for {module_name} not found")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(hasattr(module, '__version__'), "Module should have __version__ attribute")
        except Exception as e:
            self.fail(f"Import using importlib failed: {e}")


if __name__ == "__main__":
    unittest.main()
