# TrainLens Architecture

## Data Flow

```text
customer_support_tickets.csv
            |
            v
   Data quality validator
            |
            +--> quality report
            |
            v
      DuckDB analytics
            |
            +--> processed CSV reports
            |
            v
      AI labeler
       /          \
  Groq batches   rule fallback
       \          /
        labeled_tickets.csv
                |
                v
          Review queue
                |
                +--> review_results.csv
                |
                v
        Evaluation metrics
                |
                +--> evaluation_report.json
                |
                v
         Streamlit dashboard
```

## Design Decisions

- Quality validation runs before labeling so bad input is visible before model results are trusted.
- The LLM path is optional. A local keyword classifier keeps the pipeline usable when an API key is missing or a free-tier limit is reached.
- Each labeled record includes `label_run_id`, `labeled_at`, `label_model`, and `label_method` for traceability.
- Low-confidence records go to review instead of being treated as reliable predictions.
- Evaluation uses the original category and sentiment fields as ground truth for this synthetic benchmark.

## Production Changes

A production deployment should replace simulated review with an authenticated reviewer service, store runs in a database, add model and prompt versioning, and monitor data drift over time.
