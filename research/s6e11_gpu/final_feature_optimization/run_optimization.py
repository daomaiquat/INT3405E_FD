"""Train variants of XGB-D03 for the FINAL feature comparison on RTX 4060."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan_runs"
REFERENCE = PLAN / "results" / "full" / "XGB-D03"
RESULTS = HERE / "results"
sys.path.insert(0, str(PLAN))
from run_plan import Experiment, PlanRun, read_json, sha256, write_json


ORIGINAL_WIDTH = 109
VARIANTS = {
    "PAIR-TE": ("p1", "p2"),
    "RARE-TE": (),
    "PAIR-RARE-TE": ("p1", "p2"),
}
WIDTHS = {"PAIR-TE": 117, "RARE-TE": 111, "PAIR-RARE-TE": 119}
RARE_VARIANTS = {"RARE-TE", "PAIR-RARE-TE"}


class OptimizationRun(PlanRun):
    def __init__(self, output, smoke):
        super().__init__(output, smoke=smoke)
        original = read_json(REFERENCE / "config.json")
        self.original = Experiment(**{**original, "groups": tuple(original["groups"])})
        self.rare = None
        if not smoke:
            metrics = read_json(REFERENCE / "metrics.json")
            for key, value in self.manifest.items():
                if metrics[key] != value:
                    raise ValueError(f"Original XGB-D03 has a different {key}")
            with np.load(REFERENCE / "predictions.npz", allow_pickle=False) as forecast:
                for key, values in (("ids", self.ids), ("test_ids", self.test_ids), ("fold", self.folds)):
                    np.testing.assert_array_equal(forecast[key], values)

    def variant_manifest(self, suffix):
        files = (Path(__file__), *(PLAN / name for name in ("run_plan.py", "features.py", "models.py")))
        if suffix in RARE_VARIANTS:
            files = (*files, HERE / "rare_key_te.py")
        return {
            "reference": self.original.name,
            "expected_input_width": WIDTHS[suffix],
            "added_pair_groups": list(VARIANTS[suffix]),
            "added_rare_columns": ["amount_usd", "account_balance_usd"] if suffix in RARE_VARIANTS else [],
            "baseline_config_sha256": sha256(REFERENCE / "config.json"),
            "source_sha256": {path.name: sha256(path) for path in files},
        }

    def matrices(self, experiment, partition, queries):
        suffix = experiment.name.removeprefix("XGB-D03-")
        matrices = super().matrices(experiment, partition, queries)
        if suffix in RARE_VARIANTS:
            additions = self.rare.make(partition, queries, self.y, experiment.encoder_seed)
            matrices = tuple(np.column_stack((base, extra)) for base, extra in zip(matrices, additions, strict=True))
        if not self.smoke and any(matrix.shape[1] != WIDTHS[suffix] for matrix in matrices):
            raise ValueError(f"Unexpected feature width for {experiment.name}")
        return matrices

    def fit_fold(self, experiment, part, queries):
        predictions, details = super().fit_fold(experiment, part, queries)
        if not self.smoke:
            width = WIDTHS[experiment.name.removeprefix("XGB-D03-")]
            details.update(feature_count=width, extra_feature_count=width - ORIGINAL_WIDTH)
        return predictions, details

    def train_variant(self, suffix):
        if suffix in RARE_VARIANTS and self.rare is None:
            from rare_key_te import RareKeyEncoder

            self.rare = RareKeyEncoder(self.train, self.test)
        experiment = replace(
            self.original,
            name=f"XGB-D03-{suffix}",
            groups=(*self.original.groups, *VARIANTS[suffix]),
            parent=self.original.name,
        )
        directory = self.output / experiment.name
        directory.mkdir(exist_ok=True)
        feature_path = directory / "feature_manifest.json"
        feature_manifest = self.variant_manifest(suffix)
        if feature_path.is_file() and read_json(feature_path) != feature_manifest:
            raise ValueError(f"Feature implementation changed since the saved folds: {feature_path}")
        if not feature_path.is_file():
            write_json(feature_path, feature_manifest)
        self.run(experiment)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true", help="Two reduced folds with a small tree cap")
    args = parser.parse_args()
    run = OptimizationRun(RESULTS / ("smoke" if args.smoke else "full"), args.smoke)
    for suffix in dict.fromkeys(args.variants):
        run.train_variant(suffix)


if __name__ == "__main__":
    main()
