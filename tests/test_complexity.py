"""Guard the single-variable comparisons required for step 6."""

import copy
import unittest
from pathlib import Path

from src.artifacts import read_json
from src.complexity import CANDIDATES, candidate_config


class ComplexityTests(unittest.TestCase):
    def test_treatments_only_change_leaves_or_depth(self):
        original = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
        before = copy.deepcopy(original)
        for family, control in (("lightgbm", "S6-L15"), ("xgboost", "S6-X03")):
            for name, (_, group, parameter, _) in CANDIDATES.items():
                if group != family:
                    continue
                treatment = candidate_config(original, name)
                reference = candidate_config(original, control)
                treatment["baselines"][family].pop(parameter)
                reference["baselines"][family].pop(parameter)
                self.assertEqual(treatment, reference)
        self.assertEqual(original, before)

    def test_smoke_changes_budget_but_preserves_feature_and_split_config(self):
        original = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
        for name, (_, family, _, _) in CANDIDATES.items():
            full = candidate_config(original, name)
            smoke = candidate_config(original, name, True)
            self.assertEqual(smoke["baselines"][family]["max_iterations"], 30)
            smoke["baselines"][family]["max_iterations"] = full["baselines"][family]["max_iterations"]
            smoke["baselines"]["early_stopping"]["patience"] = full["baselines"]["early_stopping"]["patience"]
            self.assertEqual(smoke, full)
            self.assertEqual(full["baselines"]["lightgbm"]["metric"], "auc")


if __name__ == "__main__":
    unittest.main()
