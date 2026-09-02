import unittest

import numpy as np

from rankalign.metrics import subject_rank_predictions, top_k_predictions


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.subjects = np.asarray(["s1"] * 8 + ["s2"] * 8)
        self.scores = np.asarray([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5] * 2)

    def test_top_four_per_subject(self):
        prediction = top_k_predictions(self.scores, self.subjects, top_k=4)
        for subject in np.unique(self.subjects):
            self.assertEqual(4, int(prediction[self.subjects == subject].sum()))

    def test_oof_ranking_uses_only_positive_count(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1] * 2)
        prediction = subject_rank_predictions(labels, self.scores, self.subjects)
        np.testing.assert_array_equal(labels, prediction)


if __name__ == "__main__":
    unittest.main()
