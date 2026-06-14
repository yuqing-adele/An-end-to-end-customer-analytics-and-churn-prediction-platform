-- ============================================================================
-- Category Profit Margins
--
-- Joins order_items -> inventory_items (for landed cost) -> products (for
-- category/department) to compute revenue, cost, gross profit, margin %,
-- and return rate per product category. Cancelled items are excluded since
-- they never generated revenue or consumed inventory cost.
-- ============================================================================
CREATE OR REPLACE TABLE category_profit_margins AS
SELECT
    p.department,
    p.category,
    COUNT(*)                                                              AS items_sold,
    ROUND(SUM(oi.sale_price), 2)                                          AS revenue,
    ROUND(SUM(ii.cost), 2)                                                AS total_cost,
    ROUND(SUM(oi.sale_price) - SUM(ii.cost), 2)                           AS gross_profit,
    ROUND(100.0 * (SUM(oi.sale_price) - SUM(ii.cost)) / SUM(oi.sale_price), 2) AS margin_pct,
    SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END)               AS returned_items,
    ROUND(100.0 * SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END) / COUNT(*), 2) AS return_rate_pct,
    ROUND(AVG(oi.sale_price), 2)                                          AS avg_sale_price,
    ROUND(AVG(p.retail_price - p.cost), 2)                                AS avg_unit_margin
FROM order_items oi
JOIN inventory_items ii ON ii.id = oi.inventory_item_id
JOIN products p          ON p.id = oi.product_id
WHERE oi.status != 'Cancelled'
GROUP BY p.department, p.category
ORDER BY revenue DESC;
