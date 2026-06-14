"""
Load the raw theLook eCommerce CSVs into a DuckDB database.

- Creates one table per CSV file.
- Normalizes any timestamp-with-timezone columns to naive UTC TIMESTAMP so
  downstream SQL (DATE_DIFF, DATE_TRUNC, etc.) behaves consistently.
"""
import os
import time
import duckdb

from config import DB_PATH, DATA_DIR, CSV_FILES


def load_table(con, table_name, csv_path):
    view = f"_raw_{table_name}"
    posix_path = csv_path.replace("\\", "/")
    con.execute(
        f"CREATE OR REPLACE VIEW {view} AS "
        f"SELECT * FROM read_csv_auto('{posix_path}', sample_size=-1, all_varchar=False)"
    )

    columns = con.execute(f"DESCRIBE {view}").fetchall()

    select_parts = []
    for col_name, col_type, *_ in columns:
        if "TIMESTAMP" in col_type:
            select_parts.append(f'CAST("{col_name}" AS TIMESTAMP) AS "{col_name}"')
        else:
            select_parts.append(f'"{col_name}"')

    select_clause = ", ".join(select_parts)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT {select_clause} FROM {view}")
    con.execute(f"DROP VIEW {view}")

    n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return n


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    con = duckdb.connect(DB_PATH)
    con.execute("SET TimeZone='UTC';")

    print(f"Loading raw CSVs from {DATA_DIR} into {DB_PATH}\n")
    for table_name, csv_file in CSV_FILES.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        t0 = time.time()
        n_rows = load_table(con, table_name, csv_path)
        print(f"  {table_name:<20} {n_rows:>10,} rows  ({time.time() - t0:5.1f}s)")

    print("\nSchema summary:")
    for table_name in CSV_FILES:
        cols = con.execute(f"DESCRIBE {table_name}").fetchall()
        col_str = ", ".join(f"{c[0]}:{c[1]}" for c in cols)
        print(f"  {table_name}: {col_str}")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
