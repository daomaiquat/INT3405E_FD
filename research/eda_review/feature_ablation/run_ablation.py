"""Test the two EDA features separately on the locked S6E11 XGB-D03 pipeline."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = PROJECT / "s6e11_gpu" / "plan_runs"
BASELINE = PLAN / "results" / "full" / "XGB-D03"
RESULTS = HERE / "results"
sys.path.insert(0, str(PLAN))
from run_plan import Experiment, PlanRun, read_json, sha256, write_json


BASE_COLUMNS = 109
HOUR_BLOCK_SIZE = 6
HOUR_BLOCKS = 24 // HOUR_BLOCK_SIZE
TRANSACTION_COUNT_EDGES = (3, 6, 10)
TRANSACTION_RECENCY_EDGES = (0.25, 1, 6, 24)
MIN_CONTEXT_ROWS = 250
VARIANTS = {
    "CHANNEL-6H": {
        "feature": "channel × hour_block_6h",
        "representation": "one_hot",
        "hour_block_size": HOUR_BLOCK_SIZE,
        "fit_vocab_on_reference_only": True,
        "unknown_channel": "all zeros",
        "expected_columns": 20,
    },
    "VELOCITY-RESIDUAL": {
        "feature": "velocity_score minus cohort train median",
        "representation": "continuous",
        "count_edges": list(TRANSACTION_COUNT_EDGES),
        "recency_edges_hours": list(TRANSACTION_RECENCY_EDGES),
        "minimum_cohort_rows": MIN_CONTEXT_ROWS,
        "small_cohort_fallback": "global median of the active reference rows",
        "expected_columns": 1,
    },
}
EXP_NAMES = {f"XGB-D03-{suffix}": suffix for suffix in VARIANTS}


class AdditionalFeatures:
    def __init__(self, train, test):
        def values(column):
            return np.concatenate((train[column].to_numpy(), test[column].to_numpy()))

        self.channel = values("channel")
        hour = values("time_of_day_hour")
        if np.any((hour < 0) | (hour >= 24)) or np.any(hour % 1 != 0):
            raise ValueError("time_of_day_hour must contain whole hours from 0 to 23")
        self.hour_block = hour.astype(np.intp) // HOUR_BLOCK_SIZE
        counts = np.digitize(values("txn_count_last_24h"), TRANSACTION_COUNT_EDGES)
        recency = np.digitize(values("hours_since_last_txn"), TRANSACTION_RECENCY_EDGES)
        self.n_contexts = (len(TRANSACTION_COUNT_EDGES) + 1) * (len(TRANSACTION_RECENCY_EDGES) + 1)
        self.context = counts * (len(TRANSACTION_RECENCY_EDGES) + 1) + recency
        self.velocity = values("velocity_score").astype(np.float64)

    def channel_hour(self, reference, queries):
        categories = np.unique(self.channel[reference])
        width = len(categories) * HOUR_BLOCKS
        matrices = []
        for rows in (reference, *queries):
            codes = pd.Categorical(self.channel[rows], categories=categories).codes
            known = codes >= 0
            columns = codes[known].astype(np.intp) * HOUR_BLOCKS + self.hour_block[rows[known]]
            matrix = np.zeros((len(rows), width), dtype=np.float32)
            matrix[np.flatnonzero(known), columns] = 1
            matrices.append(matrix)
        return matrices

    def velocity_residual(self, reference, queries):
        contexts = self.context[reference]
        velocity = self.velocity[reference]
        median = np.full(self.n_contexts, np.median(velocity), dtype=np.float64)
        for group in np.flatnonzero(np.bincount(contexts, minlength=self.n_contexts) >= MIN_CONTEXT_ROWS):
            median[group] = np.median(velocity[contexts == group])
        return [
            (self.velocity[rows] - median[self.context[rows]]).astype(np.float32)[:, None]
            for rows in (reference, *queries)
        ]


class FeatureAblationRun(PlanRun):
    def __init__(self, output, smoke):
        super().__init__(output, smoke=smoke)
        self.additions = AdditionalFeatures(self.train, self.test)
        self.baseline_specification = read_json(BASELINE / "config.json")
        self.baseline_specification["groups"] = tuple(self.baseline_specification["groups"])
        self.feature_manifest = {
            "baseline": "XGB-D03", "baseline_columns": BASE_COLUMNS, "baseline_config_sha256": sha256(BASELINE / "config.json"),
            "training": "early stopping on A/E and fit on all outer T with the selected number of trees",
            "runner_sha256": sha256(Path(__file__)),
            "pipeline_hashes": {name: sha256(PLAN / name) for name in ("run_plan.py", "features.py", "models.py")},
            "variants": {name: VARIANTS[suffix] for name, suffix in EXP_NAMES.items()},
        }
        if not smoke:
            base = read_json(BASELINE / "metrics.json")
            for field in ("fold_hash", "early_stop_hash", "inputs", "smoke"):
                if base[field] != self.manifest[field]:
                    raise ValueError(f"XGB-D03 has a different {field}")
            with np.load(BASELINE / "predictions.npz", allow_pickle=False) as expected:
                for field, observed in (("ids", self.ids), ("test_ids", self.test_ids), ("fold", self.folds)):
                    np.testing.assert_array_equal(expected[field], observed)
            self.feature_manifest["baseline_predictions_sha256"] = sha256(BASELINE / "predictions.npz")
        manifest_path = self.output / "feature_manifest.json"
        if manifest_path.is_file() and read_json(manifest_path) != self.feature_manifest:
            raise ValueError(f"Feature source or configuration changed in {self.output}")
        if not manifest_path.is_file():
            write_json(manifest_path, self.feature_manifest)

    def matrices(self, experiment, partition, queries):
        original = super().matrices(experiment, partition, queries)
        name = EXP_NAMES[experiment.name]
        if name == "CHANNEL-6H":
            extra = self.additions.channel_hour(partition, queries)
        else:
            extra = self.additions.velocity_residual(partition, queries)
        if not self.smoke:
            if any(matrix.shape[1] != BASE_COLUMNS for matrix in original):
                raise ValueError("Base feature width differs from XGB-D03")
            if any(matrix.shape[1] != VARIANTS[name]["expected_columns"] for matrix in extra):
                raise ValueError(f"Unexpected feature width for {experiment.name}")
        if not all(np.isfinite(matrix).all() for matrix in extra):
            raise ValueError(f"Non-finite added features for {experiment.name}")
        return tuple(np.column_stack((base, additional)) for base, additional in zip(original, extra, strict=True))

    def fit_fold(self, experiment, part, queries):
        predictions, detail = super().fit_fold(experiment, part, queries)
        if not self.smoke:
            extra = VARIANTS[EXP_NAMES[experiment.name]]["expected_columns"]
            detail.update(base_feature_count=BASE_COLUMNS, extra_feature_count=extra, feature_count=BASE_COLUMNS + extra)
        return predictions, detail


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true", help="Use two reduced folds and a small tree cap")
    args = parser.parse_args()
    run = FeatureAblationRun(RESULTS / ("smoke" if args.smoke else "full"), args.smoke)
    original = Experiment(**run.baseline_specification)
    for suffix in dict.fromkeys(args.variants):
        experiment = replace(original, name=f"XGB-D03-{suffix}", parent=original.name)
        run.run(experiment)


if __name__ == "__main__":
    main()
