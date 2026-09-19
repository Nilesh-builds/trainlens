"""Tests for TrainLens review queue."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
from src.review.review_queue import ReviewQueue, ReviewItem, simulate_reviews


def test_review_queue_pending_count():
    items = [
        ReviewItem("CONV-00001", "test", "billing", "neutral", 0.5),
        ReviewItem("CONV-00002", "test", "refund", "negative", 0.3),
    ]
    queue = ReviewQueue(items)
    assert queue.pending_count == 2
    assert queue.total_count == 2


def test_review_queue_submit_review():
    items = [ReviewItem("CONV-00001", "test", "billing", "neutral", 0.5)]
    queue = ReviewQueue(items)
    queue.submit_review("CONV-00001", "billing", "positive", "analyst")
    assert queue.reviewed_count == 1
    assert queue.pending_count == 0


def test_review_queue_agreement_stats():
    items = [
        ReviewItem("CONV-00001", "test", "billing", "neutral", 0.5),
        ReviewItem("CONV-00002", "test", "refund", "negative", 0.3),
    ]
    queue = ReviewQueue(items)
    queue.submit_review("CONV-00001", "billing", "neutral", "analyst")
    queue.submit_review("CONV-00002", "billing", "neutral", "analyst")
    stats = queue.get_agreement_stats()
    assert stats["category_agreement"] == 50.0
    assert stats["sentiment_agreement"] == 50.0
    assert stats["total_reviewed"] == 2


def test_simulate_reviews():
    items = [
        ReviewItem(f"CONV-{i:05d}", "test", "billing", "neutral", 0.5)
        for i in range(20)
    ]
    queue = ReviewQueue(items)
    queue = simulate_reviews(queue, review_ratio=0.5)
    assert queue.reviewed_count > 0
    assert queue.pending_count < 20
