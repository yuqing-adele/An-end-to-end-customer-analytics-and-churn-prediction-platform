"""
One-off export of the small, pre-aggregated tables the dashboard reads, so
dashboard/app.py can run from a self-contained data snapshot instead of the
full (300+ MB) DuckDB warehouse.

Run after the pipeline (python run_pipeline.py) to (re)generate
dashboard/data/*.parquet.

Usage:  python export_dashboard_data.py
"""
import os

import duckdb

from config import DB_PATH

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard", "data")

QUERIES = {
    "monthly_kpis": "SELECT * FROM monthly_kpis ORDER BY month",
    "mom_cohort_retention": "SELECT * FROM mom_cohort_retention ORDER BY cohort_month, month_number",
    "category_profit_margins": "SELECT * FROM category_profit_margins ORDER BY revenue DESC",
    "customer_rfm": "SELECT * FROM customer_rfm",
    "customer_segments": "SELECT user_id, cluster, segment_name FROM customer_segments",
    "churn_predictions": "SELECT user_id, churn_probability, risk_category FROM churn_predictions",
    "churn_feature_importance": "SELECT * FROM churn_feature_importance ORDER BY importance DESC",
}

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH, read_only=True)
    total_bytes = 0
    for name, sql in QUERIES.items():
        df = con.execute(sql).df()
        path = os.path.join(OUT_DIR, f"{name}.parquet")
        df.to_parquet(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        total_bytes += os.path.getsize(path)
        print(f"{name}: {len(df):,} rows, {len(df.columns)} cols -> {path} ({size_kb:.1f} KB)")
    print(f"\nTotal snapshot size: {total_bytes / 1024:.1f} KB")
