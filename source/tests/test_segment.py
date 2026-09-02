import unittest

import numpy as np
import torch

from rankalign.model import EEGNetLite
from train_segment import boundary_indices, standardize_fold


class SegmentTests(unittest.TestCase):
    def test_model_output_shape(self):
        model = EEGNetLite()
        output = model(torch.zeros(2, 1, 30, 2500))
        self.assertEqual((2, 2), tuple(output.shape))

    def test_fold_standardization_uses_training_statistics(self):
        train = np.arange(4 * 2 * 5, dtype=np.float32).reshape(4, 2, 5)
        valid = np.full((2, 2, 5), 100, dtype=np.float32)
        train_scaled, valid_scaled = standardize_fold(train, valid)
        np.testing.assert_allclose(train_scaled.mean(axis=(0, 2)), 0, atol=1e-6)
        self.assertGreater(float(valid_scaled.mean()), 1)

    def test_boundary_window_is_subject_local(self):
        subjects = np.asarray(["a"] * 8 + ["b"] * 8)
        scores = np.tile(np.arange(8, dtype=np.float32), 2)
        selected = boundary_indices(scores, subjects, window=2)
        self.assertEqual(8, len(selected))
        self.assertEqual(4, int((selected < 8).sum()))
        self.assertEqual(4, int((selected >= 8).sum()))


if __name__ == "__main__":
    unittest.main()
