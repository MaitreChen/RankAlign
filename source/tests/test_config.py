import unittest
from pathlib import Path

from rankalign.config import load_config


class ConfigTests(unittest.TestCase):
    def test_release_config(self):
        config = load_config(Path("configs/rankalign.json"))
        self.assertEqual(3, len(config["fusion"]))
        self.assertAlmostEqual(1.0, sum(config["fusion"].values()))
        self.assertGreater(len(config["segment"]["seeds"]), 1)


if __name__ == "__main__":
    unittest.main()
