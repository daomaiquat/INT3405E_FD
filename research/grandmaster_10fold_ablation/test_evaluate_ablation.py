from pathlib import Path
import json
import time
import unittest

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from evaluate_ablation import evaluate_variants


MODELS = (
    "02e_SpGLM",
    "02f_DeGLM",
    "03a_OffLGB",
    "03b_OffXGB",
    "03c_OffCAT",
    "03d_StXGB",
    "04_EBM",
)


class EvaluateVariantsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = Path(__file__).resolve().parents[1] / "Model tham khảo" / "Grandmaster" / "notebook.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        cls.namespace = {
            "np": np,
            "stats": stats,
            "minimize": minimize,
            "expit": expit,
            "log_expit": log_expit,
            "log_loss": log_loss,
            "StratifiedKFold": StratifiedKFold,
            "time": time,
            "LOCKED_MODELS": MODELS,
            "LOGIT_EPS": 1e-10,
            "META_INNER_FOLDS": 2,
        }
        for index in (7, 9):
            source = "from __future__ import annotations\n" + "".join(cells[index]["source"])
            exec(compile(source, f"{notebook}:cell{index + 1}", "exec"), cls.namespace)

    def setUp(self):
        self.y_dev = np.tile([0, 1], 8)
        self.folds = [(i, tr, va) for i, (tr, va) in enumerate(
            StratifiedKFold(2).split(self.y_dev, self.y_dev)
        )]
        self.y_holdout = np.array([0, 0, 1, 1])
        self.baseline_oof = {name: np.zeros(16) for name in MODELS}
        self.baseline_holdout = {name: np.zeros(4) for name in MODELS}
        candidate_oof = np.zeros((16, len(MODELS)))
        candidate_oof[:, 3] = 2 * self.y_dev - 1
        candidate_holdout = np.zeros((4, len(MODELS)))
        candidate_holdout[:, 3] = [-2, -1, 1, 2]
        self.candidate_oof = {"residual_te": candidate_oof}
        self.candidate_holdout = {"residual_te": candidate_holdout}

    def evaluate(self, labels, otp):
        return evaluate_variants(
            self.namespace, self.baseline_oof, self.baseline_holdout,
            self.candidate_oof, self.candidate_holdout, self.y_dev,
            self.folds, labels, otp,
        )

    def test_holdout_metrics_compare_candidate_with_baseline(self):
        metrics, probabilities = self.evaluate(self.y_holdout, np.array([False, False, True, True]))
        baseline = metrics["baseline"]
        candidate = metrics["variants"]["residual_te"]
        np.testing.assert_allclose(probabilities["baseline"], [0.5] * 4, atol=1e-10)
        self.assertEqual(baseline["roc_auc"], 0.5)
        self.assertEqual(candidate["roc_auc"], 1.0)
        self.assertAlmostEqual(candidate["delta_roc_auc"], 0.5)
        self.assertGreater(candidate["ci95_low"], 0.0)
        self.assertLess(candidate["log_loss"], baseline["log_loss"])
        self.assertEqual(baseline["otp"]["n"], 2)
        self.assertEqual(baseline["otp"]["fraud"], 2)
        self.assertAlmostEqual(baseline["otp"]["predicted_fraud"], 1.0)
        self.assertAlmostEqual(baseline["otp"]["log_loss"], 0.6931471805599453)

    def test_holdout_labels_do_not_change_predictions(self):
        no_otp = np.zeros(len(self.y_holdout), dtype=bool)
        _, original = self.evaluate(self.y_holdout, no_otp)
        metrics, changed = self.evaluate(1 - self.y_holdout, no_otp)
        for name, scores in original.items():
            np.testing.assert_array_equal(scores, changed[name])
        self.assertEqual(metrics["variants"]["residual_te"]["roc_auc"], 0.0)
        self.assertEqual(metrics["baseline"]["otp"]["n"], 0)
        self.assertIsNone(metrics["baseline"]["otp"]["log_loss"])


if __name__ == "__main__":
    unittest.main()
