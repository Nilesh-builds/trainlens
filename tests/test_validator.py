"""Tests for TrainLens data quality validator."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
from src.data_quality.validator import DatasetValidator, run_validation


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "conversation_id": [f"CONV-{i:05d}" for i in range(100)],
        "customer_message": ["Test message"] * 100,
        "category": ["billing"] * 80 + ["technical_support"] * 20,
        "sentiment": ["positive"] * 50 + ["neutral"] * 30 + ["negative"] * 20,
        "channel": ["email"] * 40 + ["chat"] * 30 + ["phone"] * 20 + ["social_media"] * 10,
        "priority": ["low"] * 30 + ["medium"] * 40 + ["high"] * 20 + ["critical"] * 10,
        "customer_satisfaction": [4] * 50 + [3] * 30 + [2] * 20,
        "created_at": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "conversation_id": ["CONV-00001", "CONV-00002", "CONV-00002", "CONV-00004"],
        "customer_message": ["Test", "", "Test", "Test"],
        "category": ["billing", "invalid_cat", "billing", ""],
        "sentiment": ["positive", "happy", "neutral", "negative"],
        "channel": ["email", "unknown_channel", "chat", "phone"],
        "priority": ["low", "medium", "high", "critical"],
        "customer_satisfaction": [4, 99, 3, 5],
        "created_at": ["2024-01-01", "2024-01-02", "invalid-date", "2024-01-04"],
    })


def test_validator_passes_clean_data(sample_df):
    validator = DatasetValidator()
    report = validator.validate(sample_df)
    assert report.overall_score > 0.95
    assert report.failed_checks <= 1


def test_validator_detects_duplicates(dirty_df):
    validator = DatasetValidator()
    report = validator.validate(dirty_df)
    unique_check = [c for c in report.checks if c.check_id == "UNIQ-CONV_ID"][0]
    assert not unique_check.passed
    assert unique_check.affected_rows > 0


def test_validator_detects_invalid_categories(dirty_df):
    validator = DatasetValidator()
    report = validator.validate(dirty_df)
    cat_check = [c for c in report.checks if c.check_id == "VALID-CATEGORY"][0]
    assert not cat_check.passed


def test_validator_detects_missing_messages(dirty_df):
    validator = DatasetValidator()
    report = validator.validate(dirty_df)
    msg_check = [c for c in report.checks if c.check_id == "COMPL-CUSTOMER_MESSAGE"][0]
    assert not msg_check.passed


def test_quality_report_serialization(sample_df):
    validator = DatasetValidator()
    report = validator.validate(sample_df)
    report_dict = report.to_dict()
    assert "total_checks" in report_dict
    assert "overall_score" in report_dict
    assert isinstance(report_dict["checks"], list)


def test_run_validation_with_real_data():
    data_path = ROOT / "data" / "raw" / "customer_support_tickets.csv"
    if data_path.exists():
        result = run_validation(data_path)
        assert "overall_score" in result
        assert result["overall_score"] > 0
        assert result["total_checks"] > 0
