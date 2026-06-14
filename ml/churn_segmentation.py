"""
ML pipeline for the Customer Analytics & Churn Prediction Platform.

1. Pulls a per-customer feature table (RFM + behavioral signals) from DuckDB.
2. K-Means customer segmentation on (recency, frequency, monetary).
3. Random Forest churn classifier (target = customer_rfm.is_churned),
   trained on non-recency behavioral/demographic features.
4. Writes results back to DuckDB as:
     - customer_segments         (per-customer cluster + segment name)
     - churn_predictions         (per-customer churn probability + risk tier)
     - churn_feature_importance  (RF feature importances, for the dashboard)
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ANALYSIS_DATE, DB_PATH

FEATURE_SQL = """
WITH item_agg AS (
    SELECT
        o.user_id,
        COUNT(*)                                                AS total_items,
        SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END) AS returned_items,
        COUNT(DISTINCT p.category)                              AS num_categories,
        COUNT(DISTINCT p.department)                            AS num_departments
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN products p     ON p.id = oi.product_id
    WHERE o.status != 'Cancelled'
    GROUP BY o.user_id
)
SELECT
    r.user_id, r.customer_name, r.email, r.age, r.gender, r.country, r.state,
    r.traffic_source, r.signup_at, r.first_order_at, r.last_order_at,
    r.frequency, r.monetary, r.recency_days, r.rfm_segment, r.is_churned,
    COALESCE(ia.total_items, 0)     AS total_items,
    COALESCE(ia.returned_items, 0)  AS returned_items,
    COALESCE(ia.num_categories, 0)  AS num_categories,
    COALESCE(ia.num_departments, 0) AS num_departments,
    DATE_DIFF('day', r.signup_at, TIMESTAMP '{ANALYSIS_DATE}') AS tenure_days,
    CASE WHEN r.frequency > 0 THEN ROUND(r.monetary / r.frequency, 2) ELSE 0 END AS avg_order_value,
    CASE WHEN COALESCE(ia.total_items, 0) > 0
         THEN ROUND(1.0 * ia.returned_items / ia.total_items, 3) ELSE 0 END AS return_rate,
    CASE WHEN r.frequency > 0 THEN DATE_DIFF('day', r.signup_at, r.first_order_at) ELSE NULL END AS days_to_first_order
FROM customer_rfm r
LEFT JOIN item_agg ia ON ia.user_id = r.user_id
"""

# Segment name sets per cluster count k, ordered from "best" (highest
# recency/frequency/monetary value score) to "worst".
SEGMENT_NAMES_BY_K = {
    3: ["Champions", "Loyal Customers", "Lost / Inactive"],
    4: ["Champions", "Loyal Customers", "At Risk", "Lost / Inactive"],
    5: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost / Inactive"],
    6: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating", "Lost / Inactive"],
}


def run_segmentation(df):
    """K-Means clustering on log-scaled RFM, with k chosen via silhouette score."""
    print("\n=== K-Means Customer Segmentation (RFM) ===")
    rfm_log = np.log1p(df[["recency_days", "frequency", "monetary"]])
    X = StandardScaler().fit_transform(rfm_log)

    from sklearn.metrics import silhouette_score

    best_k, best_score, best_labels = 4, -1, None
    for k in range(3, 7):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels, sample_size=10000, random_state=42)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    df = df.copy()
    df["cluster"] = best_labels

    profile = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
    z = (profile - profile.mean()) / profile.std()
    value_score = -z["recency_days"] + z["frequency"] + z["monetary"]
    ranking = value_score.sort_values(ascending=False).index.tolist()
    names = SEGMENT_NAMES_BY_K[best_k]
    cluster_to_name = {cid: names[i] for i, cid in enumerate(ranking)}
    df["segment_name"] = df["cluster"].map(cluster_to_name)

    print(f"\nSelected k={best_k} (silhouette={best_score:.4f})")
    summary = profile.copy()
    summary["segment_name"] = summary.index.map(cluster_to_name)
    summary["customers"] = df.groupby("cluster").size()
    print(summary.round(1))

    return df


def run_churn_model(df):
    """Random Forest churn classifier, trained on non-recency features."""
    print("\n=== Random Forest Churn Prediction ===")
    df = df.copy()

    top_countries = df["country"].value_counts().nlargest(8).index
    df["country_grp"] = df["country"].where(df["country"].isin(top_countries), "Other")
    df["days_to_first_order"] = df["days_to_first_order"].fillna(df["tenure_days"])

    numeric_features = [
        "age", "frequency", "monetary", "avg_order_value", "tenure_days",
        "total_items", "returned_items", "return_rate", "num_categories",
        "num_departments", "days_to_first_order",
    ]
    categorical_features = ["gender", "traffic_source", "country_grp"]

    X = df[numeric_features + categorical_features]
    y = df["is_churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))
    print("Confusion matrix [rows=actual, cols=predicted] (Active, Churned):")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    feature_names = numeric_features + list(
        model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    )
    importances = pd.Series(
        model.named_steps["rf"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(10).round(4))

    df["churn_probability"] = model.predict_proba(X)[:, 1].round(4)
    df["risk_category"] = pd.cut(
        df["churn_probability"], bins=[-0.01, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"]
    ).astype(str)

    return df, importances


def main():
    con = duckdb.connect(DB_PATH)
    con.execute("SET TimeZone='UTC';")

    df = con.execute(FEATURE_SQL.format(ANALYSIS_DATE=ANALYSIS_DATE)).df()
    print(f"Loaded {len(df):,} customer feature rows")

    df = run_segmentation(df)
    df, importances = run_churn_model(df)

    segments_df = df[[
        "user_id", "customer_name", "email", "country", "state", "traffic_source",
        "age", "gender", "signup_at", "recency_days", "frequency", "monetary",
        "avg_order_value", "tenure_days", "cluster", "segment_name",
    ]].copy()

    churn_df = df[[
        "user_id", "customer_name", "email", "country", "traffic_source", "segment_name",
        "recency_days", "frequency", "monetary", "return_rate", "is_churned",
        "churn_probability", "risk_category",
    ]].copy()

    importance_df = importances.reset_index()
    importance_df.columns = ["feature", "importance"]

    con.execute("CREATE OR REPLACE TABLE customer_segments AS SELECT * FROM segments_df")
    con.execute("CREATE OR REPLACE TABLE churn_predictions AS SELECT * FROM churn_df")
    con.execute("CREATE OR REPLACE TABLE churn_feature_importance AS SELECT * FROM importance_df")

    print("\n=== Wrote tables: customer_segments, churn_predictions, churn_feature_importance ===")
    print("\nSegment x Risk crosstab:")
    print(con.execute("""
        SELECT cs.segment_name, cp.risk_category, COUNT(*) AS customers
        FROM customer_segments cs
        JOIN churn_predictions cp USING (user_id)
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df())

    con.close()


if __name__ == "__main__":
    main()
