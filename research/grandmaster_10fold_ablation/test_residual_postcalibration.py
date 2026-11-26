import unittest

import numpy as np
import pandas as pd

from residual_postcalibration import fit_residual_calibrator


def make_frame(velocity, risk, auth):
    return pd.DataFrame({
        "velocity_score": velocity,
        "merchant_risk_score": risk,
        "auth_method": auth,
    })


class ResidualPostCalibrationTests(unittest.TestCase):
    def test_learns_signed_adjustments_that_improve_group_logloss(self):
        frame = make_frame(np.zeros(80), np.zeros(80), np.repeat(["OTP", "SMS"], 40))
        labels = np.r_[np.ones(36), np.zeros(4), np.ones(4), np.zeros(36)]
        base = np.zeros(80)
        model = fit_residual_calibrator(frame, base, labels, 2.0)
        updated = model.predict_logit(frame, base)

        self.assertGreater(updated[0], 0)
        self.assertLess(updated[40], 0)
        margin = (2 * labels - 1) * updated
        self.assertLess(np.logaddexp(0, -margin).mean(), np.log(2))

    def test_unknown_groups_contribute_zero_independently(self):
        values = np.repeat([1.0, 9.0], 40)
        frame = make_frame(values, values, ["OTP"] * 80)
        labels = np.r_[np.ones(36), np.zeros(4), np.ones(4), np.zeros(36)]
        model = fit_residual_calibrator(frame, np.zeros(80), labels, 2.0)

        scoring = make_frame([1.0] * 4, [1.0, 9.0, 1.0, 9.0], ["OTP", "OTP", "new", "new"])
        base = np.full(4, -1.0)
        adjustment = model.predict_logit(scoring, base) - base
        self.assertGreater(adjustment[1], 0)
        self.assertGreater(adjustment[2], 0)
        self.assertAlmostEqual(adjustment[0], adjustment[1] + adjustment[2])
        self.assertEqual(adjustment[3], 0.0)

    def test_scoring_batch_and_labels_do_not_affect_a_row(self):
        values = np.linspace(0.0, 8.0, 128)
        labels = (values > 4).astype(float)
        train = make_frame(values, values, ["OTP"] * len(values))
        model = fit_residual_calibrator(train, np.zeros(len(values)), labels, 8.0)

        row = make_frame([6.5], [6.5], ["OTP"])
        batch = pd.concat([row, make_frame([1e5] * 20, [1e6] * 20, ["absent"] * 20)], ignore_index=True)
        batch["is_fraud"] = 1
        expected = model.predict_logit(row, np.array([-0.8]))[0]
        self.assertNotEqual(expected, -0.8)
        self.assertAlmostEqual(model.predict_logit(batch, np.full(len(batch), -0.8))[0], expected)
        batch["is_fraud"] = 0
        self.assertAlmostEqual(model.predict_logit(batch, np.full(len(batch), -0.8))[0], expected)


if __name__ == "__main__":
    unittest.main()
