"""
Run the full data pipeline end-to-end:
  1. Load raw theLook eCommerce CSVs into DuckDB
  2. Run advanced SQL analytics (cohort retention, RFM, category margins, monthly KPIs)
  3. Run the ML pipeline (K-Means segmentation + Random Forest churn model)

Usage:  python run_pipeline.py
"""
import runpy
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(label, script_path, cwd):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    old_cwd = os.getcwd()
    old_path = sys.path[:]
    try:
        os.chdir(cwd)
        sys.path.insert(0, cwd)
        runpy.run_path(script_path, run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.path = old_path


if __name__ == "__main__":
    run("STEP 1/3: Load CSVs into DuckDB", os.path.join(BASE_DIR, "scripts", "load_to_duckdb.py"), os.path.join(BASE_DIR, "scripts"))
    run("STEP 2/3: Run SQL analytics (cohort, RFM, category margins, KPIs)", os.path.join(BASE_DIR, "scripts", "run_analytics.py"), os.path.join(BASE_DIR, "scripts"))
    run("STEP 3/3: Run ML pipeline (K-Means segmentation + churn model)", os.path.join(BASE_DIR, "ml", "churn_segmentation.py"), os.path.join(BASE_DIR, "ml"))
    print(f"\n{'='*70}\nPipeline complete. Launch the dashboard with:\n"
          f"  cd dashboard && streamlit run app.py\n{'='*70}")
