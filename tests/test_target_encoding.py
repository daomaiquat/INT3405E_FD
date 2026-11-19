"""Guard step-7 target-encoding math and fold boundaries."""

import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.artifacts import read_json
from src.target_encoding import (MISSING_KEY, crossfit_keys, encode_partition, fit_table,
                                 integer_keys, transform_keys)


def _frame(labels, columns=("a", "b")):
    values = np.arange(len(labels), dtype=float)
    data = {"id": np.arange(1000, 1000 + len(labels))}
    for index, column in enumerate(columns):
        data[column] = values + index * 0.37
    data["is_fraud"] = np.asarray(labels, dtype=np.int8)
    return pd.DataFrame(data)


class TargetEncodingTests(unittest.TestCase):
    def test_integer_keys_follow_original_precision_and_special_rules(self):
        numeric = integer_keys(pd.Series([1371.37, 0.07, 1.2, np.nan]), scale=100)
        self.assertEqual(numeric[0], 137137)
        self.assertEqual(numeric[1], 7)
        self.assertEqual(numeric[2], 120)
        self.assertEqual(numeric[3], MISSING_KEY)

        flags = integer_keys(pd.Series([True, False, pd.NA], dtype="boolean"), scale=100)
        self.assertEqual(flags[0], 1)
        self.assertEqual(flags[1], 0)
        self.assertEqual(flags[2], MISSING_KEY)

    def test_m_estimate_formula_and_unseen_prior(self):
        keys = np.array([1, 1, 1, 2, 2, 3], dtype=object)
        labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
        table = fit_table(keys, labels, m=2.0)
        encoded = transform_keys(np.array([1, 2, 3, 99], dtype=object), table)
        np.testing.assert_allclose(encoded, [0.6, 0.25, 2 / 3, 0.5], rtol=1e-12)
        self.assertLess(encoded[2], 1.0)

    def test_crossfit_never_uses_the_held_out_row_to_encode_itself(self):
        keys = np.array([1] * 6 + [2] * 6 + [3] * 6 + [4] * 6, dtype=object)
        labels = np.array([1, 1, 1, 0, 0, 0,
                           1, 0, 0, 0, 0, 0,
                           1, 1, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0], dtype=np.int8)
        result = crossfit_keys(keys, labels, m=2.0, n_splits=2, seed=2026)
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=2026)
        for reference, holdout in splitter.split(np.zeros(len(labels)), labels):
            prior = float(labels[reference].mean())
            for position in holdout:
                same = keys[reference] == keys[position]
                expected = (labels[reference][same].sum() + 2.0 * prior) / (same.sum() + 2.0)
                self.assertAlmostEqual(float(result["values"][position]), float(expected), places=12)
        self.assertEqual(set(result["assignments"].tolist()), {0, 1})

    def test_encode_partition_scopes_and_outer_v_labels(self):
        config = {
            "target": "is_fraud", "numeric_columns": ["a", "b"], "encoder_seed": 2026,
            "training_plan": {
                "target_encoding": {"smooth": 2.0, "inner_folds": 2, "key_scale": 100}},
        }
        labels = np.tile(np.array([1, 0, 0, 0], dtype=np.int8), 10)
        train = _frame(labels)
        test = pd.DataFrame({"id": np.arange(2000, 2010), "a": np.arange(10) * 0.5,
                             "b": np.arange(10) * 0.25})
        part = {"T": np.arange(32), "V": np.arange(32, 40),
                "A": np.arange(24), "E": np.arange(24, 32)}
        result = encode_partition(train, test, part, config, "data", "code")
        self.assertEqual([len(result["frames"][name]) for name in
                          ("T_crossfit", "V_from_T", "test_from_T", "A_crossfit", "E_from_A")],
                         [32, 8, 10, 24, 8])
        for name in ("T_crossfit", "V_from_T", "test_from_T", "A_crossfit", "E_from_A"):
            values = result["frames"][name].filter(like="te_").to_numpy()
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((values > 0) & (values < 1)).all())
        self.assertEqual(result["signatures"]["E_from_A"]["scope"], "A")
        self.assertEqual(result["signatures"]["V_from_T"]["scope"], "T")

        changed = train.copy()
        changed.iloc[np.asarray(part["V"], dtype=int), changed.columns.get_loc("is_fraud")] = (
            1 - labels[np.asarray(part["V"], dtype=int)])
        altered = encode_partition(changed, test, part, config, "data", "code")
        for name in result["frames"]:
            np.testing.assert_allclose(
                result["frames"][name].filter(like="te_").to_numpy(),
                altered["frames"][name].filter(like="te_").to_numpy(), rtol=0, atol=1e-12)

    def test_default_config_matches_step7_plan(self):
        config = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
        expected = copy.deepcopy(config["training_plan"]["target_encoding"])
        self.assertEqual(expected["column_group"], "numeric13")
        self.assertEqual(expected["smooth"], 20.0)
        self.assertEqual(expected["inner_folds"], 5)
        self.assertEqual(expected["key_scale"], 100)
        self.assertEqual(config["encoder_seed"], 2026)


if __name__ == "__main__":
    unittest.main()
