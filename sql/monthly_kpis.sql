-- ============================================================================
-- Monthly Business KPIs
--
-- Revenue, order, customer and AOV trend by month - powers the headline KPI
-- charts on the dashboard.
-- ============================================================================
CREATE OR REPLACE TABLE monthly_kpis AS
SELECT
    DATE_TRUNC('month', o.created_at)                   AS month,
    COUNT(DISTINCT o.order_id)                          AS orders,
    COUNT(DISTINCT o.user_id)                           AS customers,
    SUM(oi.sale_price)                                  AS revenue,
    ROUND(SUM(oi.sale_price) / COUNT(DISTINCT o.order_id), 2) AS aov,
    SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END)   AS returned_items,
    COUNT(*)                                            AS items_sold
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status != 'Cancelled'
GROUP BY 1
ORDER BY 1;
