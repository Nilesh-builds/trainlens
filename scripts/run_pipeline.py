"""TrainLens main pipeline — runs all stages end-to-end."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
DATA_RAW = ROOT / "data" / "raw"
DATA_LABELED = ROOT / "data" / "labeled"
DATA_PROCESSED = ROOT / "data" / "processed"


def stage_validate():
    console.print("[bold blue]Stage 1:[/] Data Quality Validation")
    from src.data_quality.validator import run_validation
    report = run_validation(DATA_RAW / "customer_support_tickets.csv")
    table = Table(title="Data Quality Report")
    table.add_column("Check", style="cyan")
    table.add_column("Dimension", style="magenta")
    table.add_column("Status", style="green" if report["overall_score"] > 90 else "yellow")
    table.add_column("Pass Rate", justify="right")
    for check in report["checks"]:
        status = "[green]PASS[/]" if check["passed"] else "[red]FAIL[/]"
        table.add_row(check["id"], check["dimension"], status, f"{check['pass_rate']}%")
    console.print(table)
    console.print(f"  Overall Score: [bold]{report['overall_score']}%[/]\n")
    return report


def stage_analytics():
    console.print("[bold blue]Stage 2:[/] Analytics Pipeline")
    from src.analytics.pipeline import SupportAnalytics
    analytics = SupportAnalytics(DATA_RAW / "customer_support_tickets.csv")
    results = analytics.export(DATA_PROCESSED)
    analytics.close()
    for name, df in results.items():
        console.print(f"  [green]OK[/] {name}: {len(df)} rows")
    console.print()
    return results


def stage_label():
    console.print("[bold blue]Stage 3:[/] AI Labeling")
    from src.ai_labeling.labeler import AILabeler
    use_llm = os.getenv("USE_LLM", "false").lower() in {"1", "true", "yes"}
    labeler = AILabeler(use_llm=use_llm, confidence_threshold=0.3)
    labeled_df = labeler.label_file(
        DATA_RAW / "customer_support_tickets.csv",
        DATA_LABELED / "labeled_tickets.csv",
    )
    n_review = labeled_df["needs_review"].sum()
    console.print(f"  [green]OK[/] Labeled {len(labeled_df)} records")
    for method, count in labeled_df["label_method"].value_counts().items():
        console.print(f"  {method}: {count} records")
    console.print(f"  [yellow]WARN[/] {n_review} flagged for review ({n_review/len(labeled_df)*100:.1f}%)\n")
    return labeled_df


def stage_review():
    console.print("[bold blue]Stage 4:[/] Human Review Simulation")
    from src.review.review_queue import ReviewQueue, simulate_reviews
    queue = ReviewQueue.from_labeled_csv(DATA_LABELED / "labeled_tickets.csv")
    console.print(f"  Queue size: {queue.pending_count} items")
    queue = simulate_reviews(queue, review_ratio=0.7)
    stats = queue.get_agreement_stats()
    queue.export(DATA_LABELED / "review_results.csv")
    console.print(f"  [green]OK[/] Reviewed {stats['total_reviewed']} items")
    console.print(f"  Category agreement: [bold]{stats['category_agreement']}%[/]")
    console.print(f"  Sentiment agreement: [bold]{stats['sentiment_agreement']}%[/]\n")
    return stats


def stage_evaluate():
    console.print("[bold blue]Stage 5:[/] Evaluation & Metrics")
    from src.evaluation.metrics import MetricsReport
    import pandas as pd
    labeled_df = pd.read_csv(DATA_LABELED / "labeled_tickets.csv")
    report_gen = MetricsReport(labeled_df)
    report = report_gen.export(DATA_PROCESSED)
    summary = report["summary"]
    console.print(f"  Category Accuracy: [bold green]{summary['category_accuracy']}%[/]")
    console.print(f"  Category F1: [bold]{summary['category_f1']}%[/]")
    console.print(f"  Category Macro F1: [bold]{summary['category_macro_f1']}%[/]")
    console.print(f"  Sentiment Accuracy: [bold green]{summary['sentiment_accuracy']}%[/]")
    console.print(f"  Sentiment F1: [bold]{summary['sentiment_f1']}%[/]")
    console.print(f"  Sentiment Macro F1: [bold]{summary['sentiment_macro_f1']}%[/]")
    console.print(f"  Needs Review: [yellow]{summary['needs_review_pct']}%[/]\n")
    return report


def main():
    console.print(Panel.fit("[bold]TrainLens Pipeline[/]", subtitle="AI Training Data Quality Platform"))
    console.print()

    stages = [
        ("validate", stage_validate),
        ("analytics", stage_analytics),
        ("label", stage_label),
        ("review", stage_review),
        ("evaluate", stage_evaluate),
    ]

    run_stages = sys.argv[1:] if len(sys.argv) > 1 else [s[0] for s in stages]

    results = {}
    for name, func in stages:
        if name in run_stages:
            results[name] = func()

    console.print(Panel.fit("[bold green]Pipeline Complete[/]", subtitle="All stages finished"))
    return results


if __name__ == "__main__":
    main()
