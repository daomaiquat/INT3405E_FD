"""Phase 8 safeguards: cache IDs, model inputs, and raw inference bundles."""

import copy
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import read_json
from src.features import feature_matrix
from src.models import fit_bundle, predict_bundle
from src.target_encoding import integer_keys, transform_keys
from src.te_models import align_te, phase8_config


CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")


def raw_fixture():
    n = 60
    data = {"id": np.arange(n)}
    data.update({c: np.arange(n) % 8 + 0.25 for c in CONFIG["numeric_columns"]})
    data.update({c: ["A", "B"] * (n // 2) for c in CONFIG["categorical_columns"]})
    data.update({c: np.zeros(n, dtype=int) for c in CONFIG["flag_columns"]})
    data["is_fraud"] = np.tile([0, 0, 1], n // 3)
    return pd.DataFrame(data)


class TEModelTests(unittest.TestCase):
    def test_cache_join_uses_ids_and_rejects_wrong_role(self):
        raw = pd.DataFrame({"id": [4, 1, 9], "a": [1, 2, 3]})
        encoded = pd.DataFrame({"id": [1, 9, 4], "te_a": [.2, .3, .1]})
        result = align_te(raw, encoded, ["a"])
        np.testing.assert_allclose(result.te_a, [.1, .2, .3])
        wrong = encoded.copy()
        wrong.loc[0, "id"] = 99
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            align_te(raw, wrong, ["a"])
        with self.assertRaises(ValueError):
            align_te(raw, pd.concat([encoded.iloc[:1]] * 3), ["a"])

    def test_bad_probabilities_rejected(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            align_te(pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [1], "te_a": [np.inf]}), ["a"])

    def test_glm_logit_and_tree_probability_keep_raw_features(self):
        raw = raw_fixture()
        for c in CONFIG["numeric_columns"]:
            raw[f"te_{c}"] = .2
        original = copy.deepcopy(CONFIG)
        glm = feature_matrix(raw, phase8_config(CONFIG, "T01"), "glm")
        tree = feature_matrix(raw, phase8_config(CONFIG, "T02"), "tree_onehot")
        np.testing.assert_allclose(glm.te_amount_usd, np.log(.2/.8))
        np.testing.assert_allclose(tree.te_amount_usd, .2)
        np.testing.assert_array_equal(glm.amount_usd, raw.amount_usd)
        self.assertIn("log_amount_usd", glm)
        self.assertIn("merchant_category", glm)
        self.assertEqual(CONFIG, original)
        self.assertNotIn("te_amount_usd", feature_matrix(raw, CONFIG, "glm"))

    def test_saved_bundle_predicts_raw_and_preserves_provided_crossfit(self):
        raw = raw_fixture()
        augmented = raw.copy()
        tables = {}
        for c in CONFIG["numeric_columns"]:
            keys = integer_keys(raw[c], 100)
            tables[c] = {"table": {k: .1 + .02*(int(k) % 7) for k in set(keys)}, "prior": .15, "m": 20.}
            augmented[f"te_{c}"] = transform_keys(keys, tables[c])
        cfg = phase8_config(CONFIG, "T01")
        model = fit_bundle("B02", augmented, cfg)
        model.update(te_tables=tables, te_scale=100)
        expected = predict_bundle(model, augmented)
        loaded = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(predict_bundle(loaded, raw), expected, atol=1e-12)
        provided = augmented.copy()
        for c in CONFIG["numeric_columns"]:
            provided[f"te_{c}"] = .8
        # Explicit train features must be used instead of full-table transforms.
        a = predict_bundle(loaded, provided)
        without_tables = dict(loaded)
        without_tables.pop("te_tables")
        np.testing.assert_allclose(a, predict_bundle(without_tables, provided), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
