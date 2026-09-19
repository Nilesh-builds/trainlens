"""Analytics pipeline using pandas and DuckDB for customer support data."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class SupportAnalytics:
    """Analytics engine for customer support datasets."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.df: pd.DataFrame | None = None
        self.conn: duckdb.DuckDBPyConnection | None = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        self.df["created_at"] = pd.to_datetime(self.df["created_at"], errors="coerce")
        self.conn = duckdb.connect(":memory:")
        self.conn.register("tickets", self.df)
        return self.df

    def category_distribution(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT category, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM tickets), 1) as pct
            FROM tickets
            WHERE category != ''
            GROUP BY category
            ORDER BY count DESC
        """).fetchdf()

    def sentiment_by_category(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT category, sentiment, COUNT(*) as count
            FROM tickets
            WHERE sentiment IN ('positive', 'neutral', 'negative')
            GROUP BY category, sentiment
            ORDER BY category, sentiment
        """).fetchdf()

    def resolution_metrics(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT resolution, COUNT(*) as count,
                   ROUND(AVG(response_time_seconds), 0) as avg_response_sec,
                   ROUND(AVG(customer_satisfaction), 2) as avg_csat
            FROM tickets
            WHERE customer_satisfaction BETWEEN 1 AND 5
            GROUP BY resolution
            ORDER BY count DESC
        """).fetchdf()

    def channel_performance(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT channel,
                   COUNT(*) as total_tickets,
                   ROUND(AVG(response_time_seconds), 0) as avg_response_sec,
                   ROUND(AVG(customer_satisfaction), 2) as avg_csat,
                   ROUND(SUM(CASE WHEN is_resolved THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as resolution_rate
            FROM tickets
            WHERE channel IN ('email', 'chat', 'phone', 'social_media')
            GROUP BY channel
            ORDER BY total_tickets DESC
        """).fetchdf()

    def agent_performance(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT agent_id,
                   COUNT(*) as tickets_handled,
                   ROUND(AVG(response_time_seconds), 0) as avg_response_sec,
                   ROUND(AVG(customer_satisfaction), 2) as avg_csat,
                   ROUND(SUM(CASE WHEN is_resolved THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as resolution_rate
            FROM tickets
            GROUP BY agent_id
            ORDER BY tickets_handled DESC
            LIMIT 20
        """).fetchdf()

    def priority_distribution(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT priority, category, COUNT(*) as count
            FROM tickets
            WHERE priority IN ('low', 'medium', 'high', 'critical')
            GROUP BY priority, category
            ORDER BY priority, count DESC
        """).fetchdf()

    def time_series(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT DATE_TRUNC('week', created_at) as week,
                   COUNT(*) as tickets,
                   ROUND(AVG(customer_satisfaction), 2) as avg_csat,
                   ROUND(AVG(response_time_seconds), 0) as avg_response
            FROM tickets
            WHERE created_at IS NOT NULL
            GROUP BY week
            ORDER BY week
        """).fetchdf()

    def hourly_patterns(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT EXTRACT(HOUR FROM created_at) as hour,
                   COUNT(*) as tickets
            FROM tickets
            WHERE created_at IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """).fetchdf()

    def run_all(self) -> dict[str, pd.DataFrame]:
        if self.df is None:
            self.load()
        return {
            "category_distribution": self.category_distribution(),
            "sentiment_by_category": self.sentiment_by_category(),
            "resolution_metrics": self.resolution_metrics(),
            "channel_performance": self.channel_performance(),
            "agent_performance": self.agent_performance(),
            "priority_distribution": self.priority_distribution(),
            "time_series": self.time_series(),
            "hourly_patterns": self.hourly_patterns(),
        }

    def close(self):
        if self.conn:
            self.conn.close()

    def export(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.run_all()
        for name, df in results.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)
        self.close()
        return results


if __name__ == "__main__":
    analytics = SupportAnalytics(Path(__file__).parent.parent / "data" / "raw" / "customer_support_tickets.csv")
    results = analytics.export(Path(__file__).parent.parent / "data" / "processed")
    for name, df in results.items():
        print(f"{name}: {len(df)} rows")
