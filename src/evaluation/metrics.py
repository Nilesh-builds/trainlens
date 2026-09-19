"""Evaluation metrics and reporting for AI labeling quality."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_mat: list[list[int]]
    labels: list[str]
    report: str
    metric_type: str


class ModelEvaluator:
    """Evaluates AI labeling quality against human-labeled ground truth."""

    def __init__(self, labeled_df: pd.DataFrame):
        self.df = labeled_df

    def _filter_valid(self, label_col: str, pred_col: str) -> tuple[pd.Series, pd.Series]:
        valid_labels = {"billing", "technical_support", "shipping", "product_inquiry", "cancellation", "refund"}
        valid_sentiments = {"positive", "neutral", "negative"}

        valid_set = valid_labels if "category" in label_col else valid_sentiments
        mask = self.df[label_col].isin(valid_set) & self.df[pred_col].isin(valid_set)
        return self.df.loc[mask, label_col], self.df.loc[mask, pred_col]

    def evaluate_category(self) -> EvaluationResult:
        y_true, y_pred = self._filter_valid("category", "predicted_category")
        if len(y_true) == 0:
            return EvaluationResult(0, 0, 0, 0, [], [], "No valid data", "category")

        labels = sorted(y_true.unique())
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(y_true, y_pred, zero_division=0)

        return EvaluationResult(acc, prec, rec, f1, cm, labels, report, "category")

    def evaluate_sentiment(self) -> EvaluationResult:
        y_true, y_pred = self._filter_valid("sentiment", "predicted_sentiment")
        if len(y_true) == 0:
            return EvaluationResult(0, 0, 0, 0, [], [], "No valid data", "sentiment")

        labels = ["negative", "neutral", "positive"]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

        return EvaluationResult(acc, prec, rec, f1, cm, labels, report, "sentiment")

    def confidence_analysis(self) -> pd.DataFrame:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels_bin = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        self.df = self.df.copy()
        self.df["confidence_bin"] = pd.cut(self.df["prediction_confidence"], bins=bins, labels=labels_bin)

        results = []
        for bin_label in labels_bin:
            bin_df = self.df[self.df["confidence_bin"] == bin_label]
            if len(bin_df) == 0:
                continue
            cat_correct = (bin_df["predicted_category"] == bin_df["category"]).mean()
            sent_correct = (bin_df["predicted_sentiment"] == bin_df["sentiment"]).mean()
            results.append({
                "confidence_bin": bin_label,
                "count": len(bin_df),
                "category_accuracy": round(cat_correct * 100, 2),
                "sentiment_accuracy": round(sent_correct * 100, 2),
            })
        return pd.DataFrame(results)

    def error_analysis(self) -> pd.DataFrame:
        errors = self.df[
            (self.df["predicted_category"] != self.df["category"]) |
            (self.df["predicted_sentiment"] != self.df["sentiment"])
        ].copy()
        errors["category_correct"] = errors["predicted_category"] == errors["category"]
        errors["sentiment_correct"] = errors["predicted_sentiment"] == errors["sentiment"]
        return errors[[
            "conversation_id", "customer_message", "category", "predicted_category",
            "sentiment", "predicted_sentiment", "prediction_confidence",
            "category_correct", "sentiment_correct",
        ]].head(20)


class MetricsReport:
    """Generates comprehensive evaluation report."""

    def __init__(self, labeled_df: pd.DataFrame):
        self.evaluator = ModelEvaluator(labeled_df)
        self.labeled_df = labeled_df

    def generate(self) -> dict:
        cat_eval = self.evaluator.evaluate_category()
        sent_eval = self.evaluator.evaluate_sentiment()
        conf_analysis = self.evaluator.confidence_analysis()
        errors = self.evaluator.error_analysis()

        return {
            "summary": {
                "total_samples": len(self.labeled_df),
                "category_accuracy": round(cat_eval.accuracy * 100, 2),
                "category_f1": round(cat_eval.f1 * 100, 2),
                "sentiment_accuracy": round(sent_eval.accuracy * 100, 2),
                "sentiment_f1": round(sent_eval.f1 * 100, 2),
                "needs_review_pct": round(
                    self.labeled_df["needs_review"].sum() / len(self.labeled_df) * 100, 2
                ),
            },
            "category_report": cat_eval.report,
            "sentiment_report": sent_eval.report,
            "category_confusion_matrix": {
                "labels": cat_eval.labels,
                "matrix": cat_eval.confusion_mat,
            },
            "sentiment_confusion_matrix": {
                "labels": sent_eval.labels,
                "matrix": sent_eval.confusion_mat,
            },
            "confidence_analysis": conf_analysis.to_dict("records"),
            "top_errors": errors.to_dict("records"),
        }

    def export(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.generate()
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        conf_df = self.evaluator.confidence_analysis()
        conf_df.to_csv(output_dir / "confidence_analysis.csv", index=False)

        errors_df = self.evaluator.error_analysis()
        errors_df.to_csv(output_dir / "error_analysis.csv", index=False)

        return report


import json

if __name__ == "__main__":
    labeled_path = Path(__file__).parent.parent / "data" / "labeled" / "labeled_tickets.csv"
    df = pd.read_csv(labeled_path)
    report_gen = MetricsReport(df)
    report = report_gen.export(Path(__file__).parent.parent / "data" / "processed")
    print(json.dumps(report["summary"], indent=2))
