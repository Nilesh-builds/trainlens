# TrainLens

**AI Training Data Quality and Evaluation Platform for Customer Support**

A production-grade data analytics and AI training pipeline that validates, labels, evaluates, and monitors customer support conversation data using free AI models and human-in-the-loop review.

## What It Does

TrainLens solves a real business problem: **how to measure and improve the quality of AI training data** in customer support operations.

```
Raw Data --> Quality Validation --> AI Labeling --> Human Review --> Evaluation --> Dashboard
```

### Pipeline Stages

| Stage | What It Does | Output |
|-------|-------------|--------|
| **1. Data Quality** | Validates completeness, validity, uniqueness, consistency, timeliness | Quality score, issue report |
| **2. Analytics** | SQL-powered analytics on categories, sentiments, channels, agents | 8 analytical reports |
| **3. AI Labeling** | Rule-based + optional LLM labeling with confidence scores | Labeled dataset with review flags |
| **4. Human Review** | Queue uncertain predictions for human correction | Agreement metrics, disagreement log |
| **5. Evaluation** | Precision, recall, F1, confusion matrices, confidence calibration | Evaluation report, error analysis |
| **6. Dashboard** | Interactive Streamlit UI with all metrics and visualizations | Real-time monitoring |

## Key Features

- **5-dimension data quality framework** (completeness, validity, uniqueness, consistency, timeliness)
- **Hybrid AI labeling** (keyword rules + free LLM APIs with fallback)
- **Confidence-based review routing** (only uncertain predictions go to humans)
- **Model evaluation** (per-category and per-sentiment precision/recall/F1)
- **Confidence calibration analysis** (does high confidence = high accuracy?)
- **Interactive dashboard** (Plotly visualizations, real-time pipeline execution)
- **17 automated tests** across all modules
- **GitHub Actions CI** (Python 3.10-3.12, linting, testing)

## Results

| Metric | Value |
|--------|-------|
| Dataset Size | 800 conversations |
| Categories | 6 (billing, tech support, shipping, product inquiry, cancellation, refund) |
| Data Quality Score | 98.8% |
| Category Accuracy | 80.9% |
| Category F1 | 80.8% |
| Needs Review Rate | 23.0% |
| Human-AI Category Agreement | 87.4% |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_pipeline.py

# Launch the dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
trainlens/
├── data/
│   ├── raw/                    # Source data
│   ├── labeled/                # AI-labeled data + review results
│   └── processed/              # Analytics outputs + evaluation reports
├── src/
│   ├── data_quality/           # Validation framework (5 dimensions)
│   ├── analytics/              # DuckDB-powered analytics pipeline
│   ├── ai_labeling/            # Rule-based + LLM labeling engine
│   ├── review/                 # Human-in-the-loop review queue
│   ├── evaluation/             # Metrics, confusion matrices, error analysis
│   └── dashboard/              # Streamlit application
├── tests/                      # 17 pytest tests
├── scripts/                    # Data generation + pipeline runner
├── .github/workflows/ci.yml   # GitHub Actions CI
├── requirements.txt
└── pyproject.toml
```

## Tech Stack

- **Data Processing:** pandas, DuckDB
- **AI Labeling:** Keyword rules + Groq/OpenRouter free APIs
- **Evaluation:** scikit-learn (precision, recall, F1, confusion matrix)
- **Visualization:** Plotly, Streamlit
- **Validation:** Pydantic, custom 5-dimension framework
- **Testing:** pytest, GitHub Actions CI

## Skills Used

This project leverages these agent skills:
- **analytics-data-analysis** — EDA pipelines, pandas workflows, visualization standards
- **analytics-engineer** — dbt-style staging/mart patterns, data modeling
- **data-quality-frameworks** — Great Expectations-style validation, data contracts
- **machine-learning** — Model evaluation, metrics, statistical analysis
- **senior-ml-engineer** — LLM integration, provider abstraction, cost tracking

## License

MIT
