# Customer Analytics & Churn Prediction Platform

**Author:** Adele

An end-to-end customer analytics and churn-prediction platform built on the
**theLook eCommerce** dataset, using a lightweight, fully local data stack:
**DuckDB + Python (pandas / scikit-learn) + Streamlit + Plotly**.

It turns ~100K customers and ~3M raw orders/items/events into:

- A SQL analytics layer (cohort retention, RFM, category margins, monthly KPIs)
- A machine-learning layer (K-Means customer segmentation + Random Forest churn model)
- An interactive, business-facing Streamlit dashboard with an auto-generated
  executive report

---

## Why this project

It's a compact demonstration of a full analytics-engineering + applied-ML
workflow: raw CSVs &rarr; a queryable warehouse &rarr; advanced SQL &rarr; ML
models &rarr; a dashboard a non-technical stakeholder could actually use to make
decisions - all runnable on a laptop with one command and no external services.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data warehouse | [DuckDB](https://duckdb.org/) - embedded OLAP SQL engine |
| ETL | Python, pandas, DuckDB `read_csv_auto` with type inference |
| Analytics SQL | CTEs, window functions (`NTILE`), `DATE_TRUNC` / `DATE_DIFF`, conditional aggregation |
| Machine learning | scikit-learn - `KMeans`, `RandomForestClassifier`, `Pipeline`, `ColumnTransformer` |
| Dashboard | Streamlit |
| Visualization | Plotly Express & Graph Objects |
| Reporting | Self-contained HTML report generation (Plotly + pandas `to_html`) |

---

## Architecture

```
ecommerce/*.csv
      |
      v
scripts/load_to_duckdb.py  --->  db/ecommerce.duckdb  (raw tables)
      |
      v
scripts/run_analytics.py  runs sql/*.sql:
      - cohort_retention.sql  -> mom_cohort_retention
      - rfm.sql               -> customer_rfm
      - category_margins.sql  -> category_profit_margins
      - monthly_kpis.sql      -> monthly_kpis
      |
      v
ml/churn_segmentation.py
      - K-Means segmentation   -> customer_segments
      - Random Forest churn    -> churn_predictions, churn_feature_importance
      |
      +-----------------------------+
      v                              v
dashboard/app.py             scripts/export_dashboard_data.py
  (Streamlit, reads                  |
   db/ecommerce.duckdb live)         v
                            dashboard/data/*.parquet  (~7MB, committed)
                                      |
                                      v
                            dashboard/app_snapshot.py
                              (Streamlit, self-contained)
```

Run the whole pipeline with `python run_pipeline.py`, then launch either
dashboard - see [Getting Started](#getting-started).

---

## Data

**theLook eCommerce** is a realistic synthetic ecommerce dataset (Google BigQuery
public dataset). Tables loaded into DuckDB:

| Table | Rows | Description |
|---|---|---|
| `users` | 100,000 | Customer profiles - demographics, signup date, traffic source |
| `orders` | 125,226 | Order headers - status, timestamps |
| `order_items` | 181,759 | Line items - sale price, status, returns |
| `inventory_items` | 490,705 | Per-unit landed cost |
| `products` | 29,120 | Product catalog - category, department, retail price, cost |
| `events` | 2,431,963 | Web / session clickstream events |
| `distribution_centers` | 10 | Warehouse locations |

Orders span **2019-01-06 to 2024-01-17**. `ANALYSIS_DATE = 2024-01-18` (the day
after the last recorded order) is used as "today" for every recency, cohort and
churn calculation, so results stay stable no matter when the dashboard is opened.

---

## SQL Analytics (`sql/`)

| Script | Output table | What it computes |
|---|---|---|
| `cohort_retention.sql` | `mom_cohort_retention` (689 rows) | Acquisition-month cohorts; % of each cohort still ordering in months 0-12 |
| `rfm.sql` | `customer_rfm` (100,000 rows) | Recency / Frequency / Monetary per customer, `NTILE(5)` quintile scoring, a rule-based RFM segment (Champions, At Risk, Cant Lose Them, Hibernating, ...) and an `is_churned` flag |
| `category_margins.sql` | `category_profit_margins` (36 rows) | Revenue, COGS, gross margin %, return rate per product category |
| `monthly_kpis.sql` | `monthly_kpis` (61 rows) | Monthly revenue, orders, active customers, average order value |

---

## Machine Learning (`ml/churn_segmentation.py`)

**Customer segmentation** - K-Means on log-scaled (recency, frequency, monetary),
standardized, with *k* chosen automatically via silhouette score (k=3 selected,
silhouette &asymp; 0.51). Clusters are ranked by a recency/frequency/monetary value
score and mapped to business-friendly names:

| Segment | Customers | Profile |
|---|---|---|
| Champions | 21,713 | Recent, frequent, high spend |
| Loyal Customers | 50,784 | Solid, regular repeat buyers |
| Lost / Inactive | 27,503 | Long-gone or never purchased |

**Churn prediction** - `RandomForestClassifier` (300 trees, `class_weight='balanced'`)
trained on tenure, order history, return behavior and demographics. **Recency is
deliberately excluded** as a feature since it defines the churn label (avoids
leakage). ROC-AUC &asymp; **0.96**.

Top predictive features:

| Feature | Importance |
|---|---|
| `tenure_days` | 0.490 |
| `days_to_first_order` | 0.264 |
| `frequency` | 0.052 |
| `monetary` | 0.045 |
| `avg_order_value` | 0.038 |

Risk tiers - Low (&lt;0.40) / Medium (0.40-0.70) / High (&gt;0.70) - split
18,077 / 29,528 / 52,395 customers.

---

## Dashboard (`dashboard/app.py` / `dashboard/app_snapshot.py`)

Two interchangeable entry points share the same 6-tab UI:

- **`app_snapshot.py`** - self-contained; reads the ~7MB pre-computed snapshot
  in `dashboard/data/*.parquet` (committed to the repo). Runs immediately after
  `pip install -r requirements.txt`, with no database and no raw data - this is
  the one to use for a quick look or to deploy to Streamlit Community Cloud.
- **`app.py`** - reads live from `db/ecommerce.duckdb`, regenerated by
  `python run_pipeline.py`. Use this when iterating on the SQL/ML layers, since
  it always reflects the current warehouse.

A 6-tab Streamlit app:

1. **Executive Summary** - narrative KPI overview, data-driven findings &
   recommendations, and a "Generate report" button that produces a downloadable,
   self-contained HTML report (KPIs, charts, each paired with a written analysis,
   plus strategic recommendations and methodology)
2. **Business Overview** - revenue/AOV trend, monthly orders & active customers,
   category profit margins, department revenue split, return-rate leaders
3. **Cohort Retention** - month-over-month retention heatmap + cohort size chart
4. **Customer Segments** - K-Means vs. RFM segments, frequency-vs-monetary
   scatter, segment profile table
5. **Churn Risk** - risk-tier breakdown, churn probability distribution, model
   feature importance, and a searchable/filterable/exportable high-risk customer
   list
6. **Geographic** - world choropleth and U.S. state choropleth, switchable
   between lifetime revenue, customer count, churn rate and high-risk customers

Every chart is paired with a plain-language "insight box" explaining what it
shows and how to read it, so the dashboard works for a business stakeholder, not
just an analyst.

---

## Key Results

| Metric | Value |
|---|---|
| Total revenue | $9.22M |
| Total orders | 106,617 |
| Customers | 100,000 (72,123 have placed at least one order) |
| Average order value | $86.52 |
| Gross margin | 51.9% |
| Churn rate (90-day inactivity) | 79.6% |
| Churn model ROC-AUC | &asymp; 0.96 |

---

## Project Structure

```
churn_platform/
├── ecommerce/                  # Raw theLook eCommerce CSVs (gitignored)
├── db/ecommerce.duckdb         # DuckDB warehouse (generated by the pipeline, gitignored)
├── scripts/
│   ├── config.py                # Shared config - paths, ANALYSIS_DATE, CHURN_RECENCY_DAYS
│   ├── load_to_duckdb.py         # CSV -> DuckDB loader (type inference + timestamp cleanup)
│   ├── run_analytics.py          # Runs sql/*.sql against the warehouse
│   └── export_dashboard_data.py  # Exports the small tables app_snapshot.py needs to dashboard/data/*.parquet
├── sql/
│   ├── cohort_retention.sql
│   ├── rfm.sql
│   ├── category_margins.sql
│   └── monthly_kpis.sql
├── ml/
│   └── churn_segmentation.py    # K-Means segmentation + Random Forest churn model
├── dashboard/
│   ├── app.py                   # Streamlit dashboard - reads db/ecommerce.duckdb live
│   ├── app_snapshot.py          # Streamlit dashboard - self-contained, reads dashboard/data/
│   └── data/*.parquet           # ~7MB pre-computed snapshot (committed)
├── run_pipeline.py               # One-command pipeline runner
└── requirements.txt
```

---

## Getting Started

### Quick start (recommended)

The dashboard's data is pre-computed and checked into `dashboard/data/*.parquet`
(~7MB), so `app_snapshot.py` runs immediately from a fresh clone - no database
and no raw data required:

```bash
pip install -r requirements.txt
cd dashboard
streamlit run app_snapshot.py
```

### Full pipeline (`app.py`)

To regenerate everything from raw data and explore via the live-warehouse
dashboard:

> `ecommerce/*.csv` and `db/` are gitignored (raw data is ~500MB and the DuckDB
> warehouse is a generated artifact). Export the seven theLook eCommerce tables
> (`distribution_centers`, `products`, `users`, `orders`, `order_items`,
> `inventory_items`, `events`) - e.g. from the `bigquery-public-data.thelook_ecommerce`
> public BigQuery dataset - as CSVs into `ecommerce/` before running the pipeline;
> it recreates `db/ecommerce.duckdb` from scratch.

```bash
pip install -r requirements.txt

# 1. Load CSVs -> DuckDB, run SQL analytics, run the ML pipeline
python run_pipeline.py

# 2. Launch the live dashboard
cd dashboard
streamlit run app.py

# 3. (optional) refresh the snapshot used by app_snapshot.py
cd ../scripts
python export_dashboard_data.py
```

---

## Skills Demonstrated

- **SQL**: CTEs, window functions (`NTILE`), date-truncated cohort analysis,
  conditional aggregation, multi-table joins across a star-like schema
- **Data engineering**: config-driven, idempotent ETL pipeline; type inference and
  timestamp normalization from raw CSV into an analytical warehouse
- **Machine learning**: unsupervised segmentation with automatic model selection
  (silhouette score), supervised classification with class-imbalance handling,
  feature importance interpretation, and avoidance of label leakage
- **Data visualization & storytelling**: multi-tab interactive dashboard, dual-axis
  and heatmap charts, auto-generated business narratives, exportable HTML reports
- **Software engineering**: modular, reproducible pipeline orchestrated by a single
  entry point, with shared configuration across the SQL, ML and dashboard layers

---

## Possible Future Enhancements

- Date-range filters on the Business Overview tab
- A per-customer "360" drill-down view (order history, RFM trajectory over time)
- In-dashboard model diagnostics (ROC curve, confusion matrix)
- Deploy `app_snapshot.py` to Streamlit Community Cloud for a live, shareable demo link
