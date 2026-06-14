"""
Run the advanced SQL analytics scripts against the DuckDB database, materializing:
  - mom_cohort_retention
  - customer_rfm
  - category_profit_margins
  - monthly_kpis
"""
import os
import duckdb

from config import DB_PATH, SQL_DIR, ANALYSIS_DATE, CHURN_RECENCY_DAYS

SQL_FILES = [
    "cohort_retention.sql",
    "rfm.sql",
    "category_margins.sql",
    "monthly_kpis.sql",
]


def main():
    con = duckdb.connect(DB_PATH)
    con.execute("SET TimeZone='UTC';")

    for fname in SQL_FILES:
        path = os.path.join(SQL_DIR, fname)
        with open(path, "r") as f:
            sql = f.read()

        sql = sql.format(ANALYSIS_DATE=ANALYSIS_DATE, CHURN_RECENCY_DAYS=CHURN_RECENCY_DAYS)
        con.execute(sql)

        # Report on the table created (assumes "CREATE OR REPLACE TABLE <name>" is first statement)
        table_name = sql.split("CREATE OR REPLACE TABLE")[1].split("AS")[0].strip()
        n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {fname:<25} -> {table_name:<25} ({n:,} rows)")

    print("\nSample: mom_cohort_retention")
    print(con.execute("SELECT * FROM mom_cohort_retention LIMIT 5").df())

    print("\nSample: customer_rfm segment counts")
    print(con.execute("SELECT rfm_segment, COUNT(*) AS customers, ROUND(AVG(monetary),2) AS avg_monetary, "
                       "ROUND(100.0*SUM(is_churned)/COUNT(*),1) AS pct_churned "
                       "FROM customer_rfm GROUP BY 1 ORDER BY customers DESC").df())

    print("\nSample: category_profit_margins")
    print(con.execute("SELECT * FROM category_profit_margins LIMIT 10").df())

    print("\nSample: monthly_kpis")
    print(con.execute("SELECT * FROM monthly_kpis ORDER BY month DESC LIMIT 5").df())

    con.close()


if __name__ == "__main__":
    main()
