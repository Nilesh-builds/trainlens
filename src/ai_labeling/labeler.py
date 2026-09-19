"""AI labeling engine using free models for customer support data."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)
_llm_unavailable = False


@dataclass
class LabelResult:
    predicted_category: str
    predicted_sentiment: str
    confidence: float
    method: str


KEYWORD_RULES = {
    "billing": {
        "keywords": ["charge", "billing", "invoice", "payment", "subscription", "plan", "price", "cost", "fee", "refund"],
        "weight": 1.0,
    },
    "technical_support": {
        "keywords": ["crash", "error", "bug", "broken", "not working", "slow", "loading", "login", "password", "app", "website"],
        "weight": 1.0,
    },
    "shipping": {
        "keywords": ["delivery", "shipping", "tracking", "order", "package", "courier", "address", "return", "label"],
        "weight": 1.0,
    },
    "product_inquiry": {
        "keywords": ["feature", "compatibility", "pricing", "available", "stock", "compare", "difference", "upgrade", "plan"],
        "weight": 0.8,
    },
    "cancellation": {
        "keywords": ["cancel", "unsubscribe", "delete", "remove", "pause", "downgrade", "stop"],
        "weight": 1.0,
    },
    "refund": {
        "keywords": ["refund", "money back", "reimburse", "credit", "charge back"],
        "weight": 1.2,
    },
}

SENTIMENT_KEYWORDS = {
    "positive": {
        "keywords": ["great", "awesome", "excellent", "thank", "love", "perfect", "amazing", "happy", "helpful", "wonderful"],
        "weight": 1.0,
    },
    "negative": {
        "keywords": ["terrible", "awful", "worst", "hate", "angry", "frustrated", "unacceptable", "disappointed", "horrible", "annoyed"],
        "weight": 1.0,
    },
}


def rule_based_label(text: str) -> LabelResult:
    """Label using keyword matching with scoring."""
    text_lower = text.lower()

    category_scores = {}
    for cat, config in KEYWORD_RULES.items():
        score = sum(1 for kw in config["keywords"] if kw in text_lower)
        category_scores[cat] = score * config["weight"]

    if max(category_scores.values()) == 0:
        predicted_category = "unknown"
        category_confidence = 0.0
    else:
        predicted_category = max(category_scores, key=category_scores.get)
        total = sum(category_scores.values())
        category_confidence = category_scores[predicted_category] / total if total > 0 else 0.0

    sentiment_scores = {}
    for sent, config in SENTIMENT_KEYWORDS.items():
        score = sum(1 for kw in config["keywords"] if kw in text_lower)
        sentiment_scores[sent] = score * config["weight"]

    if max(sentiment_scores.values()) == 0:
        predicted_sentiment = "neutral"
        sentiment_confidence = 0.5
    else:
        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        total = sum(sentiment_scores.values())
        sentiment_confidence = sentiment_scores[predicted_sentiment] / total if total > 0 else 0.5

    overall_confidence = (category_confidence + sentiment_confidence) / 2

    return LabelResult(
        predicted_category=predicted_category,
        predicted_sentiment=predicted_sentiment,
        confidence=round(overall_confidence, 3),
        method="rule_based",
    )


def llm_based_label(text: str) -> LabelResult:
    """Label using a free LLM API with fallback to rules."""
    global _llm_unavailable

    if _llm_unavailable:
        return rule_based_label(text)

    try:
        import httpx

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return rule_based_label(text)

        prompt = f"""Classify this customer support message.
Return ONLY a JSON object with these fields:
- category: one of [billing, technical_support, shipping, product_inquiry, cancellation, refund]
- sentiment: one of [positive, neutral, negative]
- confidence: a number between 0 and 1

Message: "{text[:500]}"

JSON:"""

        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
            },
            timeout=10.0,
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return LabelResult(
                    predicted_category=data.get("category", "unknown"),
                    predicted_sentiment=data.get("sentiment", "neutral"),
                    confidence=float(data.get("confidence", 0.5)),
                    method="llm",
                )
        else:
            _llm_unavailable = True
            logger.warning(
                "Groq request failed with HTTP %s (%s); using rule-based fallback",
                response.status_code,
                response.text[:180].replace("\n", " "),
            )
    except Exception as exc:
        _llm_unavailable = True
        logger.warning("Groq request failed (%s); using rule-based fallback", type(exc).__name__)

    return rule_based_label(text)


def llm_batch_label(texts: list[str]) -> list[LabelResult]:
    """Label several messages per request to reduce free-tier API usage."""
    global _llm_unavailable
    if _llm_unavailable or not os.getenv("GROQ_API_KEY"):
        return [rule_based_label(text) for text in texts]

    try:
        import httpx

        messages = "\n".join(
            f'{index}: "{text[:500].replace(chr(34), chr(39))}"'
            for index, text in enumerate(texts)
        )
        prompt = f"""Classify each customer support message below.
Return ONLY a JSON array with exactly one object per input, in the same order.
Each object must contain: category, sentiment, confidence.
Allowed category values: billing, technical_support, shipping, product_inquiry, cancellation, refund, unknown.
Allowed sentiment values: positive, neutral, negative.

Messages:
{messages}

JSON array:"""
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json={
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                # Reasoning models may spend tokens before returning the JSON array.
                "max_tokens": max(1500, len(texts) * 150),
            },
            timeout=30.0,
        )
        if response.status_code != 200:
            _llm_unavailable = True
            logger.warning("Groq batch request failed with HTTP %s; using fallback", response.status_code)
            return [rule_based_label(text) for text in texts]

        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            raise ValueError(f"LLM response was not a JSON array: {content[:220]!r}")
        data = json.loads(match.group())
        if len(data) != len(texts):
            raise ValueError(f"LLM returned {len(data)} labels for {len(texts)} messages")
        return [LabelResult(
            predicted_category=item.get("category", "unknown"),
            predicted_sentiment=item.get("sentiment", "neutral"),
            confidence=float(item.get("confidence", 0.5)),
            method="llm",
        ) for item in data]
    except Exception as exc:
        _llm_unavailable = True
        logger.warning("Groq batch request failed (%s: %s); using fallback", type(exc).__name__, exc)
        return [rule_based_label(text) for text in texts]


class AILabeler:
    """Label customer support data using hybrid rule + LLM approach."""

    def __init__(self, use_llm: bool = False, confidence_threshold: float = 0.3):
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold

    def label_text(self, text: str) -> LabelResult:
        if self.use_llm:
            return llm_based_label(text)
        return rule_based_label(text)

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df["customer_message"].fillna("").tolist()
        if self.use_llm:
            batch_size = int(os.getenv("LLM_BATCH_SIZE", "20"))
            results = []
            for start in range(0, len(texts), batch_size):
                results.extend(llm_batch_label(texts[start:start + batch_size]))
        else:
            results = [rule_based_label(text) for text in texts]
        df = df.copy()
        df["predicted_category"] = [result.predicted_category for result in results]
        df["predicted_sentiment"] = [result.predicted_sentiment for result in results]
        df["prediction_confidence"] = [result.confidence for result in results]
        df["label_method"] = [result.method for result in results]
        df["needs_review"] = df["prediction_confidence"] < self.confidence_threshold
        return df

    def label_file(self, input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(input_path)
        labeled_df = self.label_dataframe(df)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_csv(output_path, index=False)
        return labeled_df


if __name__ == "__main__":
    labeler = AILabeler(use_llm=False)
    result = labeler.label_file(
        Path(__file__).parent.parent / "data" / "raw" / "customer_support_tickets.csv",
        Path(__file__).parent.parent / "data" / "labeled" / "labeled_tickets.csv",
    )
    print(f"Labeled {len(result)} records")
    print(f"Needs review: {result['needs_review'].sum()}")
    print(result[["conversation_id", "predicted_category", "predicted_sentiment", "prediction_confidence", "needs_review"]].head(10))
