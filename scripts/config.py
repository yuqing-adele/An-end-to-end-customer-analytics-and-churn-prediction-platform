"""Shared configuration for the Customer Analytics & Churn Prediction Platform."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # churn_platform/
DATA_DIR = os.path.join(BASE_DIR, "ecommerce")
DB_PATH = os.path.join(BASE_DIR, "db", "ecommerce.duckdb")
SQL_DIR = os.path.join(BASE_DIR, "sql")

# The raw dataset's last recorded order is 2024-01-17. We treat the day after
# as "today" for all recency / cohort / churn calculations so results are stable.
ANALYSIS_DATE = "2024-01-18"

# Customers with no qualifying order in the last N days are considered churned.
CHURN_RECENCY_DAYS = 90

CSV_FILES = {
    "distribution_centers": "distribution_centers.csv",
    "products": "products.csv",
    "users": "users.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "inventory_items": "inventory_items.csv",
    "events": "events.csv",
}
