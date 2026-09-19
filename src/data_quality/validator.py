"""Data quality validation framework for customer support data."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class QualityDimension(str, Enum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"


@dataclass
class QualityCheck:
    check_id: str
    dimension: QualityDimension
    severity: Severity
    passed: bool
    message: str
    affected_rows: int = 0
    total_rows: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_rows == 0:
            return 1.0
        return (self.total_rows - self.affected_rows) / self.total_rows


@dataclass
class QualityReport:
    checks: list[QualityCheck] = field(default_factory=list)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_checks(self) -> int:
        return self.total_checks - self.passed_checks

    @property
    def overall_score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.pass_rate for c in self.checks) / len(self.checks)

    @property
    def critical_failures(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed and c.severity == Severity.CRITICAL]

    def to_dict(self) -> dict:
        return {
            "total_checks": self.total_checks,
            "passed": self.passed_checks,
            "failed": self.failed_checks,
            "overall_score": round(self.overall_score * 100, 2),
            "critical_failures": len(self.critical_failures),
            "checks": [
                {
                    "id": c.check_id,
                    "dimension": c.dimension.value,
                    "severity": c.severity.value,
                    "passed": c.passed,
                    "message": c.message,
                    "affected_rows": c.affected_rows,
                    "pass_rate": round(c.pass_rate * 100, 2),
                }
                for c in self.checks
            ],
        }


class DatasetValidator:
    """Validates customer support datasets across multiple quality dimensions."""

    VALID_CATEGORIES = {
        "billing", "technical_support", "shipping",
        "product_inquiry", "cancellation", "refund",
    }
    VALID_SENTIMENTS = {"positive", "neutral", "negative"}
    VALID_CHANNELS = {"email", "chat", "phone", "social_media"}
    VALID_PRIORITIES = {"low", "medium", "high", "critical"}

    def validate(self, df: pd.DataFrame) -> QualityReport:
        report = QualityReport()
        report.checks.extend(self._check_completeness(df))
        report.checks.extend(self._check_validity(df))
        report.checks.extend(self._check_uniqueness(df))
        report.checks.extend(self._check_consistency(df))
        report.checks.extend(self._check_timeliness(df))
        return report

    def _check_completeness(self, df: pd.DataFrame) -> list[QualityCheck]:
        checks = []
        critical_cols = ["conversation_id", "customer_message", "category"]
        for col in critical_cols:
            if col not in df.columns:
                checks.append(QualityCheck(
                    check_id=f"COMPL-{col.upper()}",
                    dimension=QualityDimension.COMPLETENESS,
                    severity=Severity.CRITICAL,
                    passed=False,
                    message=f"Missing required column: {col}",
                    total_rows=len(df),
                    affected_rows=len(df),
                ))
                continue
            missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            checks.append(QualityCheck(
                check_id=f"COMPL-{col.upper()}",
                dimension=QualityDimension.COMPLETENESS,
                severity=Severity.CRITICAL if col == "conversation_id" else Severity.WARNING,
                passed=missing == 0,
                message=f"{col}: {missing} missing/empty values",
                affected_rows=missing,
                total_rows=len(df),
            ))
        return checks

    def _check_validity(self, df: pd.DataFrame) -> list[QualityCheck]:
        checks = []
        if "category" in df.columns:
            invalid = (~df["category"].isin(self.VALID_CATEGORIES)) & (df["category"].str.strip() != "")
            invalid_count = invalid.sum()
            checks.append(QualityCheck(
                check_id="VALID-CATEGORY",
                dimension=QualityDimension.VALIDITY,
                severity=Severity.WARNING,
                passed=invalid_count == 0,
                message=f"Category: {invalid_count} invalid values",
                affected_rows=invalid_count,
                total_rows=len(df),
            ))
        if "sentiment" in df.columns:
            invalid = ~df["sentiment"].isin(self.VALID_SENTIMENTS)
            invalid_count = invalid.sum()
            checks.append(QualityCheck(
                check_id="VALID-SENTIMENT",
                dimension=QualityDimension.VALIDITY,
                severity=Severity.WARNING,
                passed=invalid_count == 0,
                message=f"Sentiment: {invalid_count} invalid values",
                affected_rows=invalid_count,
                total_rows=len(df),
            ))
        if "customer_satisfaction" in df.columns:
            out_of_range = (~df["customer_satisfaction"].between(1, 5)) & df["customer_satisfaction"].notna()
            oor_count = out_of_range.sum()
            checks.append(QualityCheck(
                check_id="VALID-CSAT",
                dimension=QualityDimension.VALIDITY,
                severity=Severity.WARNING,
                passed=oor_count == 0,
                message=f"CSAT: {oor_count} values out of 1-5 range",
                affected_rows=oor_count,
                total_rows=len(df),
            ))
        if "created_at" in df.columns:
            invalid_dates = pd.to_datetime(df["created_at"], errors="coerce").isna()
            invalid_count = invalid_dates.sum()
            checks.append(QualityCheck(
                check_id="VALID-DATE",
                dimension=QualityDimension.VALIDITY,
                severity=Severity.WARNING,
                passed=invalid_count == 0,
                message=f"Created_at: {invalid_count} invalid dates",
                affected_rows=invalid_count,
                total_rows=len(df),
            ))
        return checks

    def _check_uniqueness(self, df: pd.DataFrame) -> list[QualityCheck]:
        checks = []
        if "conversation_id" in df.columns:
            dupes = df["conversation_id"].duplicated().sum()
            checks.append(QualityCheck(
                check_id="UNIQ-CONV_ID",
                dimension=QualityDimension.UNIQUENESS,
                severity=Severity.CRITICAL,
                passed=dupes == 0,
                message=f"Conversation IDs: {dupes} duplicates found",
                affected_rows=dupes,
                total_rows=len(df),
            ))
        return checks

    def _check_consistency(self, df: pd.DataFrame) -> list[QualityCheck]:
        checks = []
        if "channel" in df.columns:
            invalid = ~df["channel"].isin(self.VALID_CHANNELS)
            invalid_count = invalid.sum()
            checks.append(QualityCheck(
                check_id="CONSIST-CHANNEL",
                dimension=QualityDimension.CONSISTENCY,
                severity=Severity.WARNING,
                passed=invalid_count == 0,
                message=f"Channel: {invalid_count} unexpected values",
                affected_rows=invalid_count,
                total_rows=len(df),
            ))
        if "priority" in df.columns:
            invalid = ~df["priority"].isin(self.VALID_PRIORITIES)
            invalid_count = invalid.sum()
            checks.append(QualityCheck(
                check_id="CONSIST-PRIORITY",
                dimension=QualityDimension.CONSISTENCY,
                severity=Severity.WARNING,
                passed=invalid_count == 0,
                message=f"Priority: {invalid_count} unexpected values",
                affected_rows=invalid_count,
                total_rows=len(df),
            ))
        return checks

    def _check_timeliness(self, df: pd.DataFrame) -> list[QualityCheck]:
        checks = []
        if "created_at" in df.columns:
            dates = pd.to_datetime(df["created_at"], errors="coerce")
            future = (dates > pd.Timestamp.now()).sum()
            checks.append(QualityCheck(
                check_id="TIME-FUTURE",
                dimension=QualityDimension.TIMELINESS,
                severity=Severity.INFO,
                passed=future == 0,
                message=f"Timeliness: {future} records with future dates",
                affected_rows=future,
                total_rows=len(df),
            ))
        return checks


def run_validation(data_path: str | Path) -> dict:
    """Run full validation pipeline and return report as dict."""
    df = pd.read_csv(data_path)
    validator = DatasetValidator()
    report = validator.validate(df)
    return report.to_dict()


if __name__ == "__main__":
    import json

    result = run_validation(Path(__file__).parent.parent / "data" / "raw" / "customer_support_tickets.csv")
    print(json.dumps(result, indent=2))
