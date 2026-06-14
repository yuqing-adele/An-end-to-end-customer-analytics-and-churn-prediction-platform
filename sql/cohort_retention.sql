-- ============================================================================
-- Month-over-Month Cohort Retention
--
-- Cohort = the calendar month of a customer's FIRST non-cancelled order
-- (acquisition cohort). For each cohort, we track the % of that cohort's
-- customers who placed at least one more (non-cancelled) order in each of
-- the following 12 months (month_number 0 = acquisition month itself).
-- ============================================================================
CREATE OR REPLACE TABLE mom_cohort_retention AS
WITH user_order_months AS (
    -- Every distinct (customer, month) in which they placed a real order
    SELECT DISTINCT
        user_id,
        DATE_TRUNC('month', created_at) AS order_month
    FROM orders
    WHERE status != 'Cancelled'
),
cohorts AS (
    -- Each customer's acquisition month = month of their first order
    SELECT
        user_id,
        MIN(order_month) AS cohort_month
    FROM user_order_months
    GROUP BY user_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
),
activity AS (
    SELECT
        c.cohort_month,
        DATE_DIFF('month', c.cohort_month, uom.order_month) AS month_number,
        COUNT(DISTINCT uom.user_id) AS active_customers
    FROM user_order_months uom
    JOIN cohorts c ON c.user_id = uom.user_id
    GROUP BY c.cohort_month, DATE_DIFF('month', c.cohort_month, uom.order_month)
)
SELECT
    a.cohort_month,
    cs.cohort_size,
    a.month_number,
    a.active_customers,
    ROUND(100.0 * a.active_customers / cs.cohort_size, 2) AS retention_pct
FROM activity a
JOIN cohort_sizes cs ON cs.cohort_month = a.cohort_month
WHERE a.month_number BETWEEN 0 AND 12
ORDER BY a.cohort_month, a.month_number;
