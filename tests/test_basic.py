import importlib.util
import sys
import unittest


class TestBasic(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1, 1)


class TestBasicImport(unittest.TestCase):
    def test_basic_import(self):
        """Test that the module can be imported without errors."""
        try:
            from drone_base._version import __version__
            self.assertIsNotNone(__version__, "Version should not be None")
            self.assertTrue(isinstance(__version__, str), "Version should be a string")
        except ImportError as e:
            self.fail(f"Import failed with error: {e}")

    def test_module_existence(self):
        """Verify the module exists in sys.modules after import."""
        try:
            import drone_base # noqa: F401
            self.assertIn('drone_base', sys.modules, "Module should be in sys.modules")
            self.assertIn('drone_base._version', sys.modules, "Version submodule should be in sys.modules")
        except ImportError as e:
            self.fail(f"Import failed with error: {e}")

    def test_importlib_approach(self):
        """Test import using importlib for more detailed error reporting."""
        module_name = 'drone_base._version'
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
