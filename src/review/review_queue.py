"""Human-in-the-loop review system for AI-labeled data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ReviewItem:
    conversation_id: str
    customer_message: str
    predicted_category: str
    predicted_sentiment: str
    prediction_confidence: float
    human_category: str = ""
    human_sentiment: str = ""
    reviewer: str = ""
    status: str = "pending"
    reviewed_at: str = ""
    notes: str = ""


class ReviewQueue:
    """Manages the human review queue for uncertain AI predictions."""

    def __init__(self, items: list[ReviewItem] | None = None):
        self.items: list[ReviewItem] = items or []

    @classmethod
    def from_labeled_csv(cls, labeled_path: str | Path, confidence_threshold: float = 0.3) -> ReviewQueue:
        df = pd.read_csv(labeled_path)
        uncertain = df[df.get("needs_review", False) == True]
        items = []
        for _, row in uncertain.iterrows():
            items.append(ReviewItem(
                conversation_id=row.get("conversation_id", ""),
                customer_message=row.get("customer_message", ""),
                predicted_category=row.get("predicted_category", ""),
                predicted_sentiment=row.get("predicted_sentiment", ""),
                prediction_confidence=row.get("prediction_confidence", 0.0),
            ))
        return cls(items)

    @property
    def pending_count(self) -> int:
        return sum(1 for item in self.items if item.status == "pending")

    @property
    def reviewed_count(self) -> int:
        return sum(1 for item in self.items if item.status == "reviewed")

    @property
    def total_count(self) -> int:
        return len(self.items)

    def get_pending(self, limit: int = 10) -> list[ReviewItem]:
        return [item for item in self.items if item.status == "pending"][:limit]

    def submit_review(
        self,
        conversation_id: str,
        human_category: str,
        human_sentiment: str,
        reviewer: str = "analyst",
        notes: str = "",
    ):
        for item in self.items:
            if item.conversation_id == conversation_id:
                item.human_category = human_category
                item.human_sentiment = human_sentiment
                item.reviewer = reviewer
                item.status = "reviewed"
                item.reviewed_at = datetime.now().isoformat()
                item.notes = notes
                break

    def get_agreement_stats(self) -> dict:
        reviewed = [item for item in self.items if item.status == "reviewed"]
        if not reviewed:
            return {"category_agreement": 0, "sentiment_agreement": 0, "total_reviewed": 0}

        cat_agree = sum(1 for i in reviewed if i.predicted_category == i.human_category)
        sent_agree = sum(1 for i in reviewed if i.predicted_sentiment == i.human_sentiment)
        total = len(reviewed)

        return {
            "category_agreement": round(cat_agree / total * 100, 2),
            "sentiment_agreement": round(sent_agree / total * 100, 2),
            "total_reviewed": total,
            "category_disagreements": total - cat_agree,
            "sentiment_disagreements": total - sent_agree,
        }

    def get_disagreements(self) -> list[dict]:
        reviewed = [item for item in self.items if item.status == "reviewed"]
        disagreements = []
        for item in reviewed:
            if item.predicted_category != item.human_category or item.predicted_sentiment != item.human_sentiment:
                disagreements.append({
                    "conversation_id": item.conversation_id,
                    "message": item.customer_message[:100],
                    "predicted_category": item.predicted_category,
                    "human_category": item.human_category,
                    "predicted_sentiment": item.predicted_sentiment,
                    "human_sentiment": item.human_sentiment,
                    "confidence": item.prediction_confidence,
                })
        return disagreements

    def export(self, output_path: str | Path):
        records = []
        for item in self.items:
            records.append({
                "conversation_id": item.conversation_id,
                "customer_message": item.customer_message,
                "predicted_category": item.predicted_category,
                "predicted_sentiment": item.predicted_sentiment,
                "prediction_confidence": item.prediction_confidence,
                "human_category": item.human_category,
                "human_sentiment": item.human_sentiment,
                "reviewer": item.reviewer,
                "status": item.status,
                "reviewed_at": item.reviewed_at,
                "notes": item.notes,
            })
        df = pd.DataFrame(records)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


def simulate_reviews(queue: ReviewQueue, review_ratio: float = 0.7) -> ReviewQueue:
    """Simulate human reviews for demo purposes."""
    import random

    pending = queue.get_pending(limit=len(queue.items))
    n_to_review = int(len(pending) * review_ratio)

    for item in pending[:n_to_review]:
        category_match = random.random() < 0.85
        sentiment_match = random.random() < 0.80

        human_cat = item.predicted_category if category_match else random.choice(
            ["billing", "technical_support", "shipping", "product_inquiry", "cancellation", "refund"]
        )
        human_sent = item.predicted_sentiment if sentiment_match else random.choice(
            ["positive", "neutral", "negative"]
        )

        queue.submit_review(
            conversation_id=item.conversation_id,
            human_category=human_cat,
            human_sentiment=human_sent,
            reviewer="simulated_analyst",
        )

    return queue


if __name__ == "__main__":
    labeled_path = Path(__file__).parent.parent / "data" / "labeled" / "labeled_tickets.csv"
    queue = ReviewQueue.from_labeled_csv(labeled_path)
    print(f"Review queue: {queue.pending_count} pending items")

    queue = simulate_reviews(queue)
    stats = queue.get_agreement_stats()
    print(f"After simulated reviews: {stats}")

    queue.export(Path(__file__).parent.parent / "data" / "labeled" / "review_results.csv")
