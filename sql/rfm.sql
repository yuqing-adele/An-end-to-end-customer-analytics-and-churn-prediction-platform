-- ============================================================================
-- RFM (Recency, Frequency, Monetary) Analysis
--
-- For every customer (including those who never ordered):
--   recency_days  = days since their last non-cancelled order
--                    (or since signup, if they never ordered), as of {ANALYSIS_DATE}
--   frequency     = number of distinct non-cancelled orders
--   monetary      = lifetime revenue (sum of order_item sale_price)
--
-- Each metric is scored 1 (worst) - 5 (best) via NTILE quintiles, combined
-- into an rfm_total, and mapped to a business-friendly rfm_segment.
-- is_churned flags customers with no order in the last {CHURN_RECENCY_DAYS} days.
-- ============================================================================
CREATE OR REPLACE TABLE customer_rfm AS
WITH order_agg AS (
    SELECT
        o.user_id,
        MIN(o.created_at) AS first_order_at,
        MAX(o.created_at) AS last_order_at,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(oi.sale_price) AS monetary
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.status != 'Cancelled'
    GROUP BY o.user_id
),
base AS (
    SELECT
        u.id AS user_id,
        u.first_name || ' ' || u.last_name AS customer_name,
        u.email,
        u.age,
        u.gender,
        u.country,
        u.state,
        u.traffic_source,
        u.created_at AS signup_at,
        oa.first_order_at,
        oa.last_order_at,
        COALESCE(oa.frequency, 0) AS frequency,
        ROUND(COALESCE(oa.monetary, 0.0), 2) AS monetary,
        DATE_DIFF('day', COALESCE(oa.last_order_at, u.created_at), TIMESTAMP '{ANALYSIS_DATE}') AS recency_days
    FROM users u
    LEFT JOIN order_agg oa ON oa.user_id = u.id
),
scored AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC)     AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC)      AS m_score
    FROM base
)
SELECT
    *,
    (r_score + f_score + m_score) AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'New Customers'
        WHEN r_score >= 3 AND m_score >= 4                  THEN 'Big Spenders'
        WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 3 THEN 'Cant Lose Them'
        WHEN r_score <= 2 AND f_score >= 3                  THEN 'At Risk'
        WHEN frequency = 0                                  THEN 'Lost (Never Purchased)'
        WHEN r_score <= 2 AND f_score <= 2                  THEN 'Hibernating'
        ELSE 'Need Attention'
    END AS rfm_segment,
    CASE WHEN recency_days > {CHURN_RECENCY_DAYS} THEN 1 ELSE 0 END AS is_churned
FROM scored;
