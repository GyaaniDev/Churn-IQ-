import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import (  # noqa: E402
    RAW_NUMERIC_FIELDS,
    build_feature_frame,
    default_customer,
    holdout_evaluation,
    load_artifacts,
    load_datasets,
    predict_customer,
    score_batch,
)


class ChurnModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df, cls.model_df = load_datasets(PROJECT_ROOT / "data")
        cls.model, cls.scaler, cls.feature_columns = load_artifacts(PROJECT_ROOT / "artifacts")

    def test_saved_model_and_scaler_are_aligned(self):
        expected = list(self.scaler.feature_names_in_)
        self.assertEqual(self.model.n_features_in_, len(expected))
        self.assertEqual(len(expected), 91)

    def test_feature_builder_matches_exported_training_row(self):
        complete = self.raw_df.dropna(
            subset=RAW_NUMERIC_FIELDS + ["Gender", "Country", "City", "Signup_Quarter"]
        )
        source_index = int(complete.index[0])
        customer = complete.loc[source_index].to_dict()
        actual = build_feature_frame(
            customer,
            self.scaler,
            self.raw_df,
            self.model_df,
            anomaly_flag=float(self.model_df.loc[source_index, "Anomaly_Flag"]),
        )
        expected_cols = list(self.scaler.feature_names_in_)
        expected = self.model_df.loc[source_index].reindex(expected_cols, fill_value=0).astype(float)
        np.testing.assert_allclose(actual.iloc[0].to_numpy(), expected.to_numpy(), rtol=1e-7, atol=1e-7)

    def test_holdout_metrics_reproduce_notebook(self):
        metrics = holdout_evaluation(self.model, self.scaler, self.model_df)
        self.assertAlmostEqual(metrics["accuracy"], 0.9211, places=4)
        self.assertAlmostEqual(metrics["precision"], 0.912446, places=5)
        self.assertAlmostEqual(metrics["recall"], 0.804152, places=5)
        self.assertAlmostEqual(metrics["f1"], 0.854883, places=5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.929632, places=5)
        self.assertEqual(metrics["confusion"].tolist(), [[6887, 223], [566, 2324]])

    def test_single_customer_prediction_is_bounded(self):
        customer = default_customer(self.raw_df, self.model_df)
        result = predict_customer(customer, self.model, self.scaler, self.raw_df, self.model_df)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIn(result["risk_level"], {"Low", "Elevated", "High"})

    def test_batch_scoring_preserves_rows_and_adds_scores(self):
        scored = score_batch(self.raw_df.head(4), self.model, self.scaler, self.raw_df, self.model_df)
        self.assertEqual(len(scored), 4)
        for column in ("Churn_Probability", "Risk_Level", "Flagged_for_Retention"):
            self.assertIn(column, scored.columns)
        self.assertTrue(scored["Churn_Probability"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
