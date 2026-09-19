"""Tests for TrainLens AI labeling engine."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
from src.ai_labeling.labeler import AILabeler, _validated_llm_result, rule_based_label


def test_rule_based_label_billing():
    result = rule_based_label("I was charged twice for my subscription")
    assert result.predicted_category == "billing"
    assert result.confidence > 0


def test_rule_based_label_technical():
    result = rule_based_label("The app keeps crashing on my phone")
    assert result.predicted_category == "technical_support"


def test_rule_based_label_shipping():
    result = rule_based_label("Where is my order tracking number")
    assert result.predicted_category == "shipping"


def test_rule_based_label_refund():
    result = rule_based_label("I want a full refund for this order")
    assert result.predicted_category == "refund"


def test_rule_based_label_cancellation():
    result = rule_based_label("I want to cancel my subscription immediately")
    assert result.predicted_category in ("cancellation", "billing")


def test_labeler_label_dataframe():
    df = pd.DataFrame({
        "conversation_id": ["CONV-00001", "CONV-00002"],
        "customer_message": [
            "I was charged twice for my subscription",
            "The app keeps crashing on my phone",
        ],
        "category": ["billing", "technical_support"],
        "sentiment": ["negative", "negative"],
    })
    labeler = AILabeler(use_llm=False, confidence_threshold=0.3)
    result = labeler.label_dataframe(df)
    assert "predicted_category" in result.columns
    assert "predicted_sentiment" in result.columns
    assert "prediction_confidence" in result.columns
    assert "needs_review" in result.columns
    assert len(result) == 2


def test_labeler_confidence_threshold():
    df = pd.DataFrame({
        "conversation_id": ["CONV-00001"],
        "customer_message": ["hello"],
        "category": ["billing"],
        "sentiment": ["neutral"],
    })
    labeler_high = AILabeler(use_llm=False, confidence_threshold=0.9)
    labeler_low = AILabeler(use_llm=False, confidence_threshold=0.0)
    result_high = labeler_high.label_dataframe(df)
    result_low = labeler_low.label_dataframe(df)
    assert result_high["needs_review"].sum() >= result_low["needs_review"].sum()


def test_llm_output_is_normalized():
    result = _validated_llm_result({
        "category": "not_a_category",
        "sentiment": "not_a_sentiment",
        "confidence": 4,
    })
    assert result.predicted_category == "unknown"
    assert result.predicted_sentiment == "neutral"
    assert result.confidence == 1.0


def test_label_lineage_is_written():
    df = pd.DataFrame({"customer_message": ["I need a refund"]})
    result = AILabeler(use_llm=False).label_dataframe(df)
    assert result["label_run_id"].str.startswith("label-").all()
    assert result["label_model"].eq("rule-based").all()
    assert result["labeled_at"].notna().all()
