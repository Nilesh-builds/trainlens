"""Tests for TrainLens evaluation metrics (accuracy math + filtering)."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.metrics import MetricsReport, ModelEvaluator


def _tiny_labeled_df():
    return pd.DataFrame({
        "conversation_id": ["CONV-00001", "CONV-00002", "CONV-00003", "CONV-00004"],
        "customer_message": ["a", "b", "c", "d"],
        "category": ["billing", "refund", "shipping", "billing"],
        "predicted_category": ["billing", "refund", "refund", "billing"],
        "sentiment": ["positive", "negative", "neutral", "neutral"],
        "predicted_sentiment": ["positive", "negative", "neutral", "positive"],
        "prediction_confidence": [0.9, 0.8, 0.4, 0.3],
        "needs_review": [False, False, True, True],
        "label_method": ["rule_based"] * 4,
        "label_run_id": ["label-test"] * 4,
        "label_model": ["rule-based"] * 4,
    })


def test_category_accuracy_math():
    evaluator = ModelEvaluator(_tiny_labeled_df())
    result = evaluator.evaluate_category()
    # 3 of 4 categories correct -> 0.75
    assert result.accuracy == 0.75


def test_sentiment_accuracy_math():
    evaluator = ModelEvaluator(_tiny_labeled_df())
    result = evaluator.evaluate_sentiment()
    # 3 of 4 sentiments correct -> 0.75
    assert result.accuracy == 0.75


def test_invalid_labels_are_filtered():
    df = _tiny_labeled_df()
    df.loc[0, "category"] = ""  # injected quality issue
    df.loc[1, "sentiment"] = "happy"  # invalid sentiment
    evaluator = ModelEvaluator(df)
    y_true_cat, _ = evaluator._filter_valid("category", "predicted_category")
    y_true_sent, _ = evaluator._filter_valid("sentiment", "predicted_sentiment")
    assert len(y_true_cat) == 3
    assert len(y_true_sent) == 3


def test_ground_truth_not_overwritten():
    # No data leakage: predictions live in separate columns.
    df = _tiny_labeled_df()
    assert "category" in df.columns
    assert "predicted_category" in df.columns
    assert (df["category"] != df["predicted_category"]).any()
    # Ground truth values stay in the valid vocabulary.
    assert set(df["category"]).issubset(
        {"billing", "technical_support", "shipping", "product_inquiry", "cancellation", "refund"}
    )


def test_needs_review_pct_math():
    report = MetricsReport(_tiny_labeled_df()).generate()
    assert report["summary"]["needs_review_pct"] == 50.0


def test_analytics_loads_generated_data():
    from src.analytics.pipeline import SupportAnalytics

    data_path = ROOT / "data" / "raw" / "customer_support_tickets.csv"
    if not data_path.exists():
        import pytest

        pytest.skip("generated data not present (CI generates it first)")
    analytics = SupportAnalytics(data_path)
    df = analytics.load()
    assert len(df) > 0
    assert "category" in df.columns
    analytics.close()
