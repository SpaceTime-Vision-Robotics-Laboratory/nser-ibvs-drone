import unittest

class TestBasic(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1, 1)

class TestBasicImport(unittest.TestCase):
    def test_basic_import(self):
        from drone_base.main.basic_file import main
        main()

if __name__ == "__main__":
    unittest.main()
