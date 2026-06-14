"""
Customer Analytics & Churn Prediction Platform
Interactive Streamlit dashboard on top of the DuckDB warehouse built from
theLook eCommerce data.

Run with:  streamlit run app.py
"""
import os
import sys
from datetime import datetime

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from config import ANALYSIS_DATE, CHURN_RECENCY_DAYS, DB_PATH

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Analytics & Churn Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1E3A8A"
ACCENT = "#0D9488"
WARNING = "#F59E0B"
DANGER = "#DC2626"
SUCCESS = "#16A34A"

SEGMENT_COLORS = {
    "Champions": "#16A34A",
    "Loyal Customers": "#2563EB",
    "Potential Loyalists": "#0D9488",
    "At Risk": "#F59E0B",
    "Hibernating": "#F97316",
    "Lost / Inactive": "#DC2626",
}
RISK_COLORS = {"Low": "#16A34A", "Medium": "#F59E0B", "High": "#DC2626"}
RFM_COLOR_SEQ = px.colors.qualitative.Bold

st.markdown(
    f"""
    <style>
    .main-header {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }}
    .main-header h1 {{ margin: 0; font-size: 1.9rem; font-weight: 700; }}
    .main-header p {{ margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }}

    [data-testid="stMetric"] {{
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid {PRIMARY};
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }}
    [data-testid="stMetricLabel"] {{ font-weight: 600; color: #475569; }}

    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {PRIMARY};
        border-bottom: 2px solid {ACCENT};
        padding-bottom: 0.3rem;
        margin: 0.6rem 0 0.8rem 0;
    }}

    .insight-box {{
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0 1.1rem 0;
        font-size: 0.92rem;
        color: #1E3A8A;
        line-height: 1.5;
    }}
    .insight-box b {{ color: {ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
@st.cache_resource
def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data
def run_query(_con, sql):
    return _con.execute(sql).df()


con = get_connection()

monthly = run_query(con, "SELECT * FROM monthly_kpis ORDER BY month")
cohort = run_query(con, "SELECT * FROM mom_cohort_retention ORDER BY cohort_month, month_number")
category_margins = run_query(con, "SELECT * FROM category_profit_margins ORDER BY revenue DESC")
rfm = run_query(con, "SELECT * FROM customer_rfm")
segments = run_query(con, "SELECT user_id, cluster, segment_name FROM customer_segments")
churn = run_query(con, "SELECT user_id, churn_probability, risk_category FROM churn_predictions")
importance = run_query(con, "SELECT * FROM churn_feature_importance ORDER BY importance DESC")

customer_df = rfm.merge(segments, on="user_id").merge(churn, on="user_id")


def fmt_currency(x):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:,.1f}K"
    return f"${x:,.0f}"


def fmt_int(x):
    return f"{x:,.0f}"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
report_generated = datetime.now().strftime("%Y-%m-%d")

st.markdown(
    f"""
    <div class="main-header">
        <h1>Customer Analytics & Churn Prediction Platform</h1>
        <p>theLook eCommerce &middot; DuckDB + scikit-learn<br/>
        Data as of {ANALYSIS_DATE} (last recorded order) &middot;
        Churn = no order in {CHURN_RECENCY_DAYS}+ days &middot;
        Report generated {report_generated}</p>
        <p style="margin-top:0.35rem; font-size:0.8rem; opacity:0.85;">Built by Adele</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar filters (apply to customer-level views: Segments & Churn Risk tabs)
# ----------------------------------------------------------------------------
st.sidebar.header("Customer Filters")
st.sidebar.caption("Applies to the Customer Segments and Churn Risk tabs.")

countries = sorted(customer_df["country"].dropna().unique().tolist())
traffic_sources = sorted(customer_df["traffic_source"].dropna().unique().tolist())
seg_names = sorted(customer_df["segment_name"].dropna().unique().tolist())
risk_levels = ["Low", "Medium", "High"]

sel_countries = st.sidebar.multiselect("Country", countries, default=[])
sel_traffic = st.sidebar.multiselect("Traffic Source", traffic_sources, default=[])
sel_segments = st.sidebar.multiselect("Behavioral Segment (K-Means)", seg_names, default=[])
sel_risk = st.sidebar.multiselect("Churn Risk", risk_levels, default=[])

filtered_df = customer_df.copy()
if sel_countries:
    filtered_df = filtered_df[filtered_df["country"].isin(sel_countries)]
if sel_traffic:
    filtered_df = filtered_df[filtered_df["traffic_source"].isin(sel_traffic)]
if sel_segments:
    filtered_df = filtered_df[filtered_df["segment_name"].isin(sel_segments)]
if sel_risk:
    filtered_df = filtered_df[filtered_df["risk_category"].isin(sel_risk)]

st.sidebar.metric("Customers in selection", f"{len(filtered_df):,} / {len(customer_df):,}")

st.sidebar.markdown("---")
st.sidebar.caption("Built by **Adele** · theLook eCommerce + DuckDB + scikit-learn + Streamlit")

# ----------------------------------------------------------------------------
# KPI row (company-wide)
# ----------------------------------------------------------------------------
total_revenue = float(monthly["revenue"].sum())
total_orders = int(monthly["orders"].sum())
total_customers = len(customer_df)
paying_customers = int((customer_df["frequency"] > 0).sum())
aov = total_revenue / total_orders if total_orders else 0
overall_margin = (
    category_margins["gross_profit"].sum() / category_margins["revenue"].sum() * 100
)
churn_rate = customer_df["is_churned"].mean() * 100
high_risk_n = int((customer_df["risk_category"] == "High").sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Revenue", fmt_currency(total_revenue))
k2.metric("Total Orders", fmt_int(total_orders))
k3.metric("Customers", fmt_int(total_customers), f"{paying_customers:,} paying")
k4.metric("Avg Order Value", f"${aov:,.2f}")
k5.metric("Gross Margin", f"{overall_margin:,.1f}%")
k6.metric("Churn Rate", f"{churn_rate:,.1f}%", f"{high_risk_n:,} high-risk", delta_color="inverse")

# ----------------------------------------------------------------------------
# Derived stats, narrative findings & downloadable report (Executive Summary)
# ----------------------------------------------------------------------------
data_start = monthly["month"].min().strftime("%b %Y")
data_end = monthly["month"].max().strftime("%b %Y")

if len(monthly) >= 6:
    recent3 = float(monthly["revenue"].tail(3).sum())
    prior3 = float(monthly["revenue"].iloc[-6:-3].sum())
    revenue_trend_pct = (recent3 - prior3) / prior3 * 100 if prior3 else 0.0
else:
    recent3 = float(monthly["revenue"].sum())
    prior3 = 0.0
    revenue_trend_pct = 0.0

top_revenue_cat = category_margins.sort_values("revenue", ascending=False).iloc[0]
best_margin_cat = category_margins.sort_values("margin_pct", ascending=False).iloc[0]
worst_margin_cat = category_margins.sort_values("margin_pct", ascending=True).iloc[0]
top_return_cat = category_margins.sort_values("return_rate_pct", ascending=False).iloc[0]

month1_retention = cohort.loc[cohort["month_number"] == 1, "retention_pct"].mean()

never_purchased_n = int((customer_df["frequency"] == 0).sum())
never_purchased_pct = never_purchased_n / total_customers * 100

champions_df = customer_df[customer_df["segment_name"] == "Champions"]
champions_ltv_share = (
    champions_df["monetary"].sum() / customer_df["monetary"].sum() * 100
    if customer_df["monetary"].sum() else 0.0
)

high_risk_value = float(customer_df.loc[customer_df["risk_category"] == "High", "monetary"].sum())
top_feature_name = importance.iloc[0]["feature"]

KEY_FINDINGS = [
    f"Revenue momentum: the most recent 3 months brought in {fmt_currency(recent3)} vs. "
    f"{fmt_currency(prior3)} in the prior 3 months ({revenue_trend_pct:+.1f}%).",
    f"\"{top_revenue_cat['category']}\" ({top_revenue_cat['department']}) is the top revenue "
    f"category at {fmt_currency(top_revenue_cat['revenue'])}, carrying a "
    f"{top_revenue_cat['margin_pct']:.1f}% gross margin.",
    f"Category margins range from {worst_margin_cat['margin_pct']:.1f}% "
    f"(\"{worst_margin_cat['category']}\") to {best_margin_cat['margin_pct']:.1f}% "
    f"(\"{best_margin_cat['category']}\").",
    f"\"{top_return_cat['category']}\" has the highest return rate at "
    f"{top_return_cat['return_rate_pct']:.1f}%, eroding realized margin on that line.",
    f"Average Month-1 cohort retention is {month1_retention:.1f}% - most customers do not place "
    f"a second order within a month of their first purchase.",
    f"{never_purchased_pct:.1f}% of registered customers ({never_purchased_n:,}) have never placed "
    f"an order; they make up the bulk of the 'Lost / Inactive' segment.",
    f"The 'Champions' segment ({len(champions_df):,} customers, "
    f"{len(champions_df) / total_customers * 100:.1f}% of the base) drives "
    f"{champions_ltv_share:.1f}% of total lifetime revenue.",
    f"Overall churn rate (no order in {CHURN_RECENCY_DAYS}+ days) is {churn_rate:.1f}%. "
    f"{high_risk_n:,} customers are flagged High risk, representing "
    f"{fmt_currency(high_risk_value)} in historical lifetime value.",
    f"The churn model's single strongest predictor is '{top_feature_name}' - how long a "
    f"customer has been registered without converting is the clearest early-warning signal.",
]

RECOMMENDATIONS = [
    "Prioritize win-back offers for the 'At Risk' and 'Hibernating' RFM segments - customers "
    "with real purchase history who haven't ordered recently are the most efficient retention "
    "spend.",
    f"Audit \"{top_return_cat['category']}\" for sizing, quality or description issues driving "
    f"its {top_return_cat['return_rate_pct']:.1f}% return rate.",
    "Use the High-Risk export on the Churn Risk tab to drive targeted retention campaigns, "
    "ranked by churn probability and lifetime value.",
    f"Improve first-purchase conversion: {never_purchased_pct:.0f}% of signups never buy. "
    f"Shortening time-to-first-order is the single biggest lever on predicted churn.",
    f"Expand marketing and shelf space for high-margin categories such as "
    f"\"{best_margin_cat['category']}\" ({best_margin_cat['margin_pct']:.1f}% margin).",
]


def build_html_report():
    """Self-contained HTML report: headline KPIs plus a narrative business
    analysis paired with each chart (revenue, category margins, segments,
    churn risk), closing with prioritized recommendations and methodology."""
    rev_fig = go.Figure()
    rev_fig.add_bar(x=monthly["month"], y=monthly["revenue"], name="Revenue", marker_color=PRIMARY)
    rev_fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["aov"], name="AOV", mode="lines+markers",
        marker_color=WARNING, yaxis="y2",
    ))
    rev_fig.update_layout(
        title="Monthly Revenue & Average Order Value",
        yaxis=dict(title="Revenue ($)", automargin=True),
        yaxis2=dict(title="AOV ($)", overlaying="y", side="right", automargin=True),
        xaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        height=440, margin=dict(t=60, b=90, l=70, r=70),
    )

    cat_fig = px.bar(
        category_margins.sort_values("revenue", ascending=True).tail(15),
        x="revenue", y="category", color="margin_pct", orientation="h",
        color_continuous_scale="RdYlGn", title="Top 15 Categories by Revenue (color = margin %)",
        labels={"revenue": "Revenue ($)", "category": "", "margin_pct": "Margin %"},
    )
    cat_fig.update_layout(
        height=540, margin=dict(t=70, b=60, l=170, r=60),
        xaxis=dict(automargin=True), yaxis=dict(automargin=True),
    )

    seg_counts_all = customer_df["segment_name"].value_counts().reset_index()
    seg_counts_all.columns = ["segment_name", "customers"]
    seg_fig = px.bar(
        seg_counts_all, x="segment_name", y="customers", color="segment_name",
        color_discrete_map=SEGMENT_COLORS, title="Customers by Behavioral Segment",
    )
    seg_fig.update_layout(
        showlegend=False, height=420, margin=dict(t=60, b=60, l=70, r=40),
        xaxis=dict(automargin=True), yaxis=dict(automargin=True),
    )

    risk_fig = px.histogram(
        customer_df, x="churn_probability", color="risk_category", nbins=30,
        color_discrete_map=RISK_COLORS, title="Churn Probability Distribution (All Customers)",
        labels={"churn_probability": "Churn Probability"},
    )
    risk_fig.update_layout(
        height=460, margin=dict(t=60, b=90, l=70, r=40),
        xaxis=dict(automargin=True), yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    rev_html = rev_fig.to_html(full_html=False, include_plotlyjs="cdn")
    cat_html = cat_fig.to_html(full_html=False, include_plotlyjs=False)
    seg_html = seg_fig.to_html(full_html=False, include_plotlyjs=False)
    risk_html = risk_fig.to_html(full_html=False, include_plotlyjs=False)

    margin_spread = best_margin_cat["margin_pct"] - worst_margin_cat["margin_pct"]
    champions_pct = len(champions_df) / total_customers * 100
    trend_word = (
        "an acceleration in demand" if revenue_trend_pct >= 0
        else "a slowdown that warrants attention"
    )

    overview_text = (
        f"Between {data_start} and {data_end}, theLook eCommerce generated "
        f"{fmt_currency(total_revenue)} in total revenue across {fmt_int(total_orders)} "
        f"completed orders placed by {fmt_int(paying_customers)} of its "
        f"{fmt_int(total_customers)} registered customers, equivalent to an average order "
        f"value of ${aov:,.2f} and a blended gross margin of {overall_margin:.1f}% after the "
        f"cost of goods sold. Under the working definition of churn applied throughout this "
        f"report - no qualifying order within {CHURN_RECENCY_DAYS} days of {ANALYSIS_DATE} - "
        f"{churn_rate:.1f}% of the customer base is currently classified as churned, and the "
        f"Random Forest model trained on this population has flagged {fmt_int(high_risk_n)} "
        f"customers as High risk, a group whose combined historical spend totals "
        f"{fmt_currency(high_risk_value)}. The sections below examine revenue performance, "
        f"category-level profitability, the composition of the customer base, the drivers "
        f"of churn risk, and the geographic distribution of revenue and churn, before "
        f"closing with a set of prioritized recommendations."
    )

    revenue_text = (
        f"The chart above plots monthly revenue as bars against the average order value "
        f"(AOV) on the secondary axis. Revenue over the three most recent months totaled "
        f"{fmt_currency(recent3)}, compared with {fmt_currency(prior3)} in the preceding "
        f"three months, a change of {revenue_trend_pct:+.1f}% that points to {trend_word}. "
        f"Because total revenue is the product of order volume and order value, a sustained "
        f"increase in AOV alongside stable order counts typically indicates that existing "
        f"customers are purchasing higher-value items or larger baskets, whereas a declining "
        f"AOV paired with rising order counts more often reflects a higher mix of discounted "
        f"or lower-priced purchases. Management should determine whether the "
        f"{revenue_trend_pct:+.1f}% change in recent revenue is being driven primarily by "
        f"order volume or by order value, since the two patterns call for different "
        f"commercial responses: the former favors continued investment in acquisition and "
        f"retention, while the latter favors a review of pricing, promotions and product mix."
    )

    category_text = (
        f"The chart above ranks the fifteen highest-revenue product categories and colors "
        f"each bar by its gross margin, calculated as gross profit - revenue less the landed "
        f"cost of goods sold - divided by revenue. \"{top_revenue_cat['category']}\" in the "
        f"{top_revenue_cat['department']} department generates the highest revenue in the "
        f"portfolio at {fmt_currency(top_revenue_cat['revenue'])}, carrying a gross margin of "
        f"{top_revenue_cat['margin_pct']:.1f}%. Margins vary considerably across the "
        f"portfolio, ranging from {worst_margin_cat['margin_pct']:.1f}% for "
        f"\"{worst_margin_cat['category']}\" to {best_margin_cat['margin_pct']:.1f}% for "
        f"\"{best_margin_cat['category']}\", a spread of {margin_spread:.1f} percentage "
        f"points that represents a meaningful opportunity for margin management through "
        f"pricing, sourcing or assortment decisions. Separately, "
        f"\"{top_return_cat['category']}\" records the highest return rate in the catalog at "
        f"{top_return_cat['return_rate_pct']:.1f}% of units sold; because every returned "
        f"item still incurs fulfillment and processing costs without generating retained "
        f"revenue, this category's effective margin is materially lower than its headline "
        f"figure suggests."
    )

    segment_text = (
        f"Customers are grouped into behavioral segments using K-Means clustering on each "
        f"customer's recency, frequency and monetary (RFM) values. The Champions segment, "
        f"comprising {fmt_int(len(champions_df))} customers, or {champions_pct:.1f}% of the "
        f"customer base, accounts for {champions_ltv_share:.1f}% of total lifetime revenue, "
        f"underscoring how concentrated the company's revenue generation is among a "
        f"relatively small group of its most engaged buyers. At the opposite end of the "
        f"spectrum, {fmt_int(never_purchased_n)} customers, or {never_purchased_pct:.1f}% of "
        f"all registrations, have never placed a single order and fall into the Lost / "
        f"Inactive segment alongside long-lapsed former buyers. This distribution points to "
        f"two distinct opportunities: converting the large pool of never-purchased "
        f"registrants into first-time buyers, and protecting the disproportionate revenue "
        f"contribution of the Champions segment through continued engagement and loyalty "
        f"investment."
    )

    churn_text = (
        f"The histogram above shows the distribution of churn probabilities produced by the "
        f"Random Forest classifier, with each customer colored by risk tier: Low (below "
        f"0.40), Medium (0.40 to 0.70) and High (above 0.70). A total of "
        f"{fmt_int(high_risk_n)} customers fall into the High risk tier, representing "
        f"historical lifetime revenue of {fmt_currency(high_risk_value)} that is at risk of "
        f"not being repeated if these customers do not return. The single most influential "
        f"factor in the model's predictions is '{top_feature_name}', which indicates that "
        f"the length of time a customer has been registered without converting into a "
        f"repeat purchaser is a stronger signal of future churn than recency, order "
        f"frequency or spend history. In practical terms, the window in which a newly "
        f"registered customer makes a second purchase is critical, and customers who pass "
        f"that window without doing so are substantially more likely to remain inactive "
        f"thereafter."
    )

    recommendations_html = "".join(f"<li>{r}</li>" for r in [
        (
            f"Launch a structured win-back program targeting the 'At Risk' and 'Cant Lose "
            f"Them' RFM segments, since these customers have demonstrated meaningful "
            f"historical spend but have not ordered recently; recovering even a modest share "
            f"of the {fmt_currency(high_risk_value)} in lifetime value currently classified "
            f"as High risk would have a direct and measurable impact on revenue."
        ),
        (
            f"Prioritize first-purchase conversion programs aimed at the "
            f"{fmt_int(never_purchased_n)} registered customers ({never_purchased_pct:.1f}% "
            f"of the base) who have never completed an order, given that days-to-first-order "
            f"is the second most important driver of churn in the model and shortening this "
            f"window is likely to reduce churn probability across the entire cohort of new "
            f"sign-ups."
        ),
        (
            f"Investigate the root causes of the {top_return_cat['return_rate_pct']:.1f}% "
            f"return rate in \"{top_return_cat['category']}\", such as sizing guidance, "
            f"product descriptions or quality control, since reducing returns in this "
            f"category would improve its realized margin without requiring any change in "
            f"price or revenue."
        ),
        (
            f"Increase marketing investment and merchandising prominence for high-margin "
            f"categories such as \"{best_margin_cat['category']}\" "
            f"({best_margin_cat['margin_pct']:.1f}% gross margin), funded in part by "
            f"reallocating budget currently directed at lower-margin categories such as "
            f"\"{worst_margin_cat['category']}\" ({worst_margin_cat['margin_pct']:.1f}% gross "
            f"margin), in order to lift the {overall_margin:.1f}% blended margin reported "
            f"across the business."
        ),
        (
            f"Protect the revenue concentration in the Champions segment, which represents "
            f"{champions_pct:.1f}% of customers but {champions_ltv_share:.1f}% of lifetime "
            f"revenue, through loyalty benefits and proactive service, since the loss of "
            f"even a small percentage of this group would have an outsized effect on total "
            f"revenue relative to its size."
        ),
    ])

    methodology_text = (
        f"All recency-, cohort- and RFM-based calculations in this report are computed as of "
        f"{ANALYSIS_DATE}, the date of the last recorded order in the source data, so that "
        f"customer recency and churn status are measured against a fixed and consistent "
        f"reference point rather than the date on which this report happens to be generated "
        f"({report_generated}). A customer is classified as churned if no qualifying order "
        f"was placed within {CHURN_RECENCY_DAYS} days of {ANALYSIS_DATE}. The churn "
        f"probability assigned to each customer is produced by a Random Forest classifier "
        f"trained on tenure, order history, return behavior and demographic attributes; "
        f"recency itself is deliberately excluded from the model's inputs because it "
        f"defines the churn label, and its inclusion would otherwise allow the model to "
        f"reproduce the label directly rather than learn from independent predictive "
        f"signals."
    )

    country_geo, state_geo = compute_geo_data()

    geo_country_fig = px.choropleth(
        country_geo, locations="country", locationmode="country names",
        color="revenue", color_continuous_scale="Blues",
        hover_name="country", labels=GEO_LABELS, hover_data=GEO_HOVER,
        title="Total Lifetime Revenue by Country",
    )
    geo_country_fig.update_layout(height=460, margin=dict(t=60, b=10, l=10, r=10))

    geo_state_fig = px.choropleth(
        state_geo, locations="state_code", locationmode="USA-states", scope="usa",
        color="revenue", color_continuous_scale="Blues",
        hover_name="state", labels=GEO_LABELS, hover_data=GEO_HOVER,
        title="Total Lifetime Revenue by U.S. State",
    )
    geo_state_fig.update_layout(height=460, margin=dict(t=60, b=10, l=10, r=10))

    geo_country_html = geo_country_fig.to_html(full_html=False, include_plotlyjs=False)
    geo_state_html = geo_state_fig.to_html(full_html=False, include_plotlyjs=False)

    top3 = country_geo.sort_values("revenue", ascending=False).head(3)
    top3_revenue = top3["revenue"].sum()
    top3_share = top3_revenue / country_geo["revenue"].sum() * 100
    top3_names = ", ".join(top3["country"].iloc[:-1]) + " and " + top3["country"].iloc[-1]

    major_markets = country_geo[country_geo["customers"] >= 1000]
    min_churn_row = major_markets.loc[major_markets["churn_rate"].idxmin()]
    max_churn_row = major_markets.loc[major_markets["churn_rate"].idxmax()]
    churn_spread = max_churn_row["churn_rate"] - min_churn_row["churn_rate"]

    us_row = country_geo.loc[country_geo["country"] == "United States"].iloc[0]
    us_customer_share = us_row["customers"] / total_customers * 100

    state_by_revenue = state_geo.sort_values("revenue", ascending=False)
    top_state, second_state = state_by_revenue.iloc[0], state_by_revenue.iloc[1]
    top_state_share = top_state["revenue"] / us_row["revenue"] * 100

    geo_text = (
        f"The customer base spans {len(country_geo)} countries, and revenue is heavily "
        f"concentrated in a handful of them: {top3_names} together contribute "
        f"{fmt_currency(top3_revenue)}, or {top3_share:.1f}% of the "
        f"{fmt_currency(country_geo['revenue'].sum())} in total lifetime revenue mapped "
        f"above, meaning retention and growth initiatives focused on these three countries "
        f"alone would influence the majority of company revenue. Despite this "
        f"concentration, churn is not a regional phenomenon: among markets with at least "
        f"1,000 customers, churn rates range narrowly from {min_churn_row['churn_rate']:.1f}% "
        f"in {min_churn_row['country']} to {max_churn_row['churn_rate']:.1f}% in "
        f"{max_churn_row['country']}, a spread of only {churn_spread:.1f} percentage points "
        f"- evidence that the churn drivers identified earlier in this report, principally "
        f"tenure and time to first order, operate consistently across geographies rather "
        f"than being specific to any one market. Within the United States, which "
        f"represents {us_customer_share:.1f}% of the global customer base and "
        f"{fmt_currency(us_row['revenue'])} in lifetime revenue, {top_state['state']} is "
        f"the leading state by revenue at {fmt_currency(top_state['revenue'])} "
        f"({top_state_share:.1f}% of U.S. revenue), followed by {second_state['state']}. "
        f"Because churn pressure is broadly uniform across regions, retention spend is "
        f"best allocated in proportion to revenue exposure - led by {top3_names} - rather "
        f"than concentrated in any single market."
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Customer Analytics &amp; Churn Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: #1E293B;
          max-width: 1100px; margin: 0 auto; padding: 2rem; }}
  h1 {{ color: {PRIMARY}; margin-bottom: 0.2rem; }}
  h2 {{ color: {PRIMARY}; border-bottom: 2px solid {ACCENT}; padding-bottom: 0.3rem; margin-top: 2.2rem; }}
  .meta {{ color: #64748B; margin-bottom: 1.5rem; }}
  p {{ line-height: 1.7; margin: 0.8rem 0 1.2rem 0; text-align: justify; }}
  .kpi-grid {{ display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 1rem 0; }}
  .kpi-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid {PRIMARY};
                border-radius: 10px; padding: 0.8rem 1.2rem; min-width: 160px; }}
  .kpi-card .label {{ font-size: 0.8rem; color: #475569; font-weight: 600; }}
  .kpi-card .value {{ font-size: 1.4rem; font-weight: 700; color: {PRIMARY}; }}
  ol {{ padding-left: 1.2rem; }}
  li {{ line-height: 1.7; margin-bottom: 0.8rem; text-align: justify; }}
  .chart-wrap {{ margin: 0.5rem 0 1.8rem 0; }}
</style>
</head>
<body>
  <h1>Customer Analytics &amp; Churn Prediction Report</h1>
  <p class="meta">theLook eCommerce &middot; Data as of {ANALYSIS_DATE} (last recorded order)
  &middot; Report generated {report_generated} &middot; Churn = no order in {CHURN_RECENCY_DAYS}+ days</p>

  <h2>Headline Metrics</h2>
  <div class="kpi-grid">
    <div class="kpi-card"><div class="label">Total Revenue</div><div class="value">{fmt_currency(total_revenue)}</div></div>
    <div class="kpi-card"><div class="label">Total Orders</div><div class="value">{fmt_int(total_orders)}</div></div>
    <div class="kpi-card"><div class="label">Customers</div><div class="value">{fmt_int(total_customers)}</div></div>
    <div class="kpi-card"><div class="label">Avg Order Value</div><div class="value">${aov:,.2f}</div></div>
    <div class="kpi-card"><div class="label">Gross Margin</div><div class="value">{overall_margin:.1f}%</div></div>
    <div class="kpi-card"><div class="label">Churn Rate</div><div class="value">{churn_rate:.1f}%</div></div>
  </div>

  <h2>Executive Overview</h2>
  <p>{overview_text}</p>

  <h2>1. Revenue and Order Trend</h2>
  <div class="chart-wrap">{rev_html}</div>
  <p>{revenue_text}</p>

  <h2>2. Category Profitability</h2>
  <div class="chart-wrap">{cat_html}</div>
  <p>{category_text}</p>

  <h2>3. Customer Segmentation</h2>
  <div class="chart-wrap">{seg_html}</div>
  <p>{segment_text}</p>

  <h2>4. Churn Risk Assessment</h2>
  <div class="chart-wrap">{risk_html}</div>
  <p>{churn_text}</p>

  <h2>5. Geographic Distribution</h2>
  <div class="chart-wrap">{geo_country_html}</div>
  <div class="chart-wrap">{geo_state_html}</div>
  <p>{geo_text}</p>

  <h2>Strategic Recommendations</h2>
  <ol>{recommendations_html}</ol>

  <h2>Methodology</h2>
  <p>{methodology_text}</p>
</body>
</html>"""


# ----------------------------------------------------------------------------
# Geographic lookups (for choropleth maps on the Geographic tab)
# ----------------------------------------------------------------------------
# A handful of countries appear under non-English aliases in the raw data;
# map them to the names Plotly's "country names" location mode recognizes.
COUNTRY_NAME_FIXES = {"Brasil": "Brazil", "Deutschland": "Germany", "España": "Spain"}

US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}

GEO_LABELS = {
    "revenue": "Lifetime Revenue ($)", "customers": "Customers",
    "churn_rate": "Churn Rate (%)", "high_risk_customers": "High-Risk Customers",
}
GEO_HOVER = {
    "customers": ":,", "revenue": ":$,.0f", "churn_rate": ":.1f", "high_risk_customers": ":,",
}


def compute_geo_data():
    """Aggregate customer_df to country- and US-state-level metrics for the choropleth maps."""
    geo_df = customer_df.copy()
    geo_df["country"] = geo_df["country"].replace(COUNTRY_NAME_FIXES)

    country_geo = geo_df.groupby("country").agg(
        customers=("user_id", "count"),
        revenue=("monetary", "sum"),
        churn_rate=("is_churned", "mean"),
        high_risk_customers=("risk_category", lambda s: (s == "High").sum()),
    ).reset_index()
    country_geo["churn_rate"] = (country_geo["churn_rate"] * 100).round(1)

    us_df = geo_df[geo_df["country"] == "United States"]
    state_geo = us_df.groupby("state").agg(
        customers=("user_id", "count"),
        revenue=("monetary", "sum"),
        churn_rate=("is_churned", "mean"),
        high_risk_customers=("risk_category", lambda s: (s == "High").sum()),
    ).reset_index()
    state_geo["churn_rate"] = (state_geo["churn_rate"] * 100).round(1)
    state_geo["state_code"] = state_geo["state"].map(US_STATE_ABBR)
    state_geo = state_geo.dropna(subset=["state_code"])

    return country_geo, state_geo


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Executive Summary", "Business Overview", "Cohort Retention", "Customer Segments",
     "Churn Risk", "Geographic"]
)

# ============================== TAB 0: EXECUTIVE SUMMARY ==============================
with tab0:
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.caption(f"Company-wide view - data as of {ANALYSIS_DATE} - report generated {report_generated}.")

    st.markdown(
        f"Between **{data_start}** and **{data_end}**, theLook eCommerce processed "
        f"**{fmt_int(total_orders)} orders** from **{fmt_int(paying_customers)}** purchasing "
        f"customers (out of **{fmt_int(total_customers)}** registered), generating "
        f"**{fmt_currency(total_revenue)}** in revenue at a blended **{overall_margin:.1f}%** "
        f"gross margin and a **${aov:,.2f}** average order value. Under a "
        f"{CHURN_RECENCY_DAYS}-day inactivity rule, the overall churn rate stands at "
        f"**{churn_rate:.1f}%**, with **{high_risk_n:,}** customers currently flagged "
        f"**High risk** of churn by the Random Forest model."
    )

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**Key Findings**")
        st.markdown("\n".join(f"- {f}" for f in KEY_FINDINGS))
    with fc2:
        st.markdown("**Recommended Actions**")
        st.markdown("\n".join(f"- {r}" for r in RECOMMENDATIONS))

    st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
    st.markdown(
        "Generate a self-contained HTML report that walks through revenue trends, category "
        "profitability, customer segmentation and churn risk - each chart paired with a "
        "written business analysis - and closes with prioritized recommendations and a "
        "methodology note. Handy for sharing with stakeholders who don't have access to "
        "this dashboard."
    )
    if st.button("Generate report"):
        st.session_state["report_html"] = build_html_report()
    if "report_html" in st.session_state:
        st.download_button(
            "Download report (HTML)",
            data=st.session_state["report_html"],
            file_name=f"churn_analytics_report_{report_generated}.html",
            mime="text/html",
        )

    with st.expander("How the figures on this page are calculated"):
        st.markdown(
            f"- **Data as of {ANALYSIS_DATE}**: every recency, cohort and RFM calculation "
            f"treats this date as \"today\" - it is the date of the last recorded order in the "
            f"source data, so customer recency and churn status are computed against a fixed, "
            f"consistent point in time rather than the date you happen to be viewing this "
            f"dashboard.\n"
            f"- **Churn** = no order placed in the {CHURN_RECENCY_DAYS} days before "
            f"{ANALYSIS_DATE} (`customer_rfm.is_churned`).\n"
            f"- **Churn probability** comes from a Random Forest model trained on tenure, order "
            f"history, returns and demographics - recency is intentionally excluded as a "
            f"feature since it defines the churn label (see the Churn Risk tab).\n"
            f"- **Behavioral segments** (Champions, Loyal Customers, ...) come from K-Means "
            f"clustering on log-scaled Recency, Frequency and Monetary values; **RFM segments** "
            f"are a separate, rule-based quintile scoring computed directly in SQL."
        )

# ============================== TAB 1: OVERVIEW ==============================
with tab1:
    st.markdown('<div class="section-title">Revenue & Order Trend</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-box">Bars show <b>monthly revenue</b>; the line shows '
        f'<b>Average Order Value (AOV)</b> on the right-hand axis. The most recent 3 months '
        f'brought in {fmt_currency(recent3)} vs. {fmt_currency(prior3)} in the prior 3 months '
        f'({revenue_trend_pct:+.1f}%). Rising AOV alongside flat order counts usually means '
        f'customers are buying more per visit; a falling AOV with rising orders often signals '
        f'more discount-driven or lower-value purchases.</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["revenue"], name="Revenue", marker_color=PRIMARY, yaxis="y1")
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["aov"], name="Avg Order Value",
            mode="lines+markers", marker_color=WARNING, yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Revenue ($)"),
        yaxis2=dict(title="AOV ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box"><b>Monthly Orders</b> tracks total order volume; '
        '<b>Monthly Active Customers</b> counts distinct customers who placed at least one '
        'order that month. Comparing the two highlights whether growth comes from more '
        'customers, or from existing customers ordering more often.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            monthly, x="month", y="orders", title="Monthly Orders",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            monthly, x="month", y="customers", title="Monthly Active Customers",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Category Profit Margins</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-box"><b>Margin % = gross profit &divide; revenue</b> for each '
        f'product category (gross profit = revenue minus cost of goods sold, before returns). '
        f'Bars are sorted by revenue and colored from red (low margin) to green (high margin). '
        f'"{best_margin_cat["category"]}" has the highest margin at '
        f'{best_margin_cat["margin_pct"]:.1f}%, while "{worst_margin_cat["category"]}" has the '
        f'lowest at {worst_margin_cat["margin_pct"]:.1f}%. Hover any bar for its department, '
        f'exact margin and return rate.</div>',
        unsafe_allow_html=True,
    )

    cat_sorted = category_margins.sort_values("revenue", ascending=True)
    fig = px.bar(
        cat_sorted,
        x="revenue", y="category", color="margin_pct", orientation="h",
        color_continuous_scale="RdYlGn",
        hover_data={"department": True, "return_rate_pct": True, "margin_pct": ":.1f"},
        labels={"revenue": "Revenue ($)", "category": "", "margin_pct": "Margin %"},
    )
    fig.update_layout(height=820, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        dept = category_margins.groupby("department")["revenue"].sum().reset_index()
        fig = px.pie(
            dept, names="department", values="revenue", hole=0.55,
            title="Revenue Share by Department",
            color_discrete_sequence=[PRIMARY, ACCENT],
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_return = category_margins.sort_values("return_rate_pct", ascending=False).head(10)
        fig = px.bar(
            top_return, x="return_rate_pct", y="category", orientation="h",
            title="Top 10 Categories by Return Rate",
            color="return_rate_pct", color_continuous_scale="OrRd",
            labels={"return_rate_pct": "Return Rate %", "category": ""},
        )
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight-box">The donut shows how revenue splits across departments. '
        f'The bar on the right ranks categories by <b>return rate</b> (returned items / items '
        f'sold) - "{top_return_cat["category"]}" is highest at '
        f'{top_return_cat["return_rate_pct"]:.1f}%. High return rates quietly erode realized '
        f'margin even when the listed margin % looks healthy.</div>',
        unsafe_allow_html=True,
    )

# ============================== TAB 2: COHORT ==============================
with tab2:
    st.markdown('<div class="section-title">Month-over-Month Cohort Retention</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-box">Each <b>row</b> is a cohort - all customers whose first '
        f'order fell in that month. Column 0 is always 100% (their first order); columns 1, 2, '
        f'3... show the % of that same cohort who placed <b>another</b> order 1, 2, 3... months '
        f'later. Reading down a column compares retention trends across cohorts; reading across '
        f'a row shows how quickly a single cohort fades. Across all cohorts, average Month-1 '
        f'retention is <b>{month1_retention:.1f}%</b> - typical for ecommerce, where most '
        f'customers are one-time buyers. Darker cells (YlGnBu scale) mean more of the cohort '
        f'returned.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Cohort = month of a customer's first order. Each cell shows the % of that "
        "cohort placing another order N months later (company-wide)."
    )

    cohort = cohort.copy()
    cohort["cohort_label"] = cohort["cohort_month"].dt.strftime("%Y-%m")
    all_cohorts = sorted(cohort["cohort_label"].unique())

    n_cohorts = st.slider("Cohorts to display (most recent N)", 6, len(all_cohorts), min(24, len(all_cohorts)))
    shown_cohorts = all_cohorts[-n_cohorts:]
    cohort_view = cohort[cohort["cohort_label"].isin(shown_cohorts)]

    pivot = cohort_view.pivot(index="cohort_label", columns="month_number", values="retention_pct")
    pivot = pivot.sort_index()

    fig = px.imshow(
        pivot, color_continuous_scale="YlGnBu", aspect="auto", text_auto=".0f",
        labels=dict(x="Months Since First Order", y="Acquisition Cohort", color="Retention %"),
    )
    fig.update_layout(height=max(420, 22 * len(pivot)), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">New customers acquired per cohort (Month 0 size) - useful '
        'context when comparing retention percentages above, since a retention % from a very '
        'small cohort can be noisy.</div>',
        unsafe_allow_html=True,
    )

    sizes = cohort_view.drop_duplicates("cohort_label")[["cohort_label", "cohort_size"]].sort_values("cohort_label")
    fig = px.bar(
        sizes, x="cohort_label", y="cohort_size", title="New Customers per Cohort (Month 0)",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ============================== TAB 3: SEGMENTS ==============================
with tab3:
    st.markdown('<div class="section-title">Customer Segments</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">Two complementary views of the same customers. '
        '<b>Behavioral Segments (K-Means)</b> are data-driven clusters discovered from each '
        "customer's Recency, Frequency and Monetary (RFM) values - ranging from Champions "
        '(recent, frequent, high spend) to Lost / Inactive (long ago, rarely or never '
        'purchased). <b>RFM Segments</b> use a traditional rule-based score: each customer is '
        'ranked 1-5 on Recency, Frequency and Monetary via SQL quintiles, and the combined '
        'score maps to a named segment (e.g. "Champions", "At Risk", "Cant Lose Them"). Use the '
        'sidebar filters to drill into a country, traffic source, segment or risk tier.</div>',
        unsafe_allow_html=True,
    )
    st.caption("Reflects the Customer Filters in the sidebar.")

    c1, c2 = st.columns(2)
    with c1:
        seg_counts = (
            filtered_df["segment_name"].value_counts().reset_index()
        )
        seg_counts.columns = ["segment_name", "customers"]
        fig = px.bar(
            seg_counts, x="segment_name", y="customers", color="segment_name",
            color_discrete_map=SEGMENT_COLORS, title="Behavioral Segments (K-Means)",
        )
        fig.update_layout(showlegend=False, height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rfm_counts = filtered_df["rfm_segment"].value_counts().reset_index()
        rfm_counts.columns = ["rfm_segment", "customers"]
        fig = px.bar(
            rfm_counts, x="customers", y="rfm_segment", orientation="h",
            color="rfm_segment", color_discrete_sequence=RFM_COLOR_SEQ,
            title="RFM Segments (rule-based SQL)",
        )
        fig.update_layout(showlegend=False, height=380, margin=dict(t=40, b=10), yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Segment Profile: Frequency vs. Monetary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">Each point is a customer: <b>x-axis</b> = number of orders '
        'placed, <b>y-axis</b> = total lifetime revenue, <b>size</b> = days since last order '
        '(bigger = longer gone), <b>color</b> = behavioral segment. Champions cluster in the '
        'top-right (frequent, high spend, small dots = recently active); Lost / Inactive '
        'cluster near the origin with large dots.</div>',
        unsafe_allow_html=True,
    )
    scatter_df = filtered_df if len(filtered_df) <= 6000 else filtered_df.sample(6000, random_state=42)
    fig = px.scatter(
        scatter_df, x="frequency", y="monetary", color="segment_name",
        size="recency_days", size_max=18, opacity=0.6,
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["customer_name", "country", "recency_days", "churn_probability"],
        labels={"frequency": "Orders (Frequency)", "monetary": "Lifetime Revenue ($)"},
    )
    fig.update_layout(height=480, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Segment Averages</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">Average recency, frequency, lifetime value, age and churn '
        'metrics per behavioral segment, sorted by lifetime value. Use this to size each '
        'segment and see how much churn risk concentrates in your highest-value customers '
        'versus your already-lapsed ones.</div>',
        unsafe_allow_html=True,
    )
    profile = (
        filtered_df.groupby("segment_name")
        .agg(
            customers=("user_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_age=("age", "mean"),
            churn_rate_pct=("is_churned", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .round(2)
        .reset_index()
    )
    profile["churn_rate_pct"] = (profile["churn_rate_pct"] * 100).round(1)
    profile["avg_churn_probability"] = (profile["avg_churn_probability"] * 100).round(1)
    st.dataframe(
        profile.sort_values("avg_monetary", ascending=False),
        hide_index=True,
        use_container_width=True,
        column_config={
            "avg_monetary": st.column_config.NumberColumn("Avg Lifetime Value", format="$%.2f"),
            "churn_rate_pct": st.column_config.NumberColumn("Churn Rate %", format="%.1f%%"),
            "avg_churn_probability": st.column_config.NumberColumn("Avg Churn Prob %", format="%.1f%%"),
        },
    )

# ============================== TAB 4: CHURN RISK ==============================
with tab4:
    st.markdown('<div class="section-title">Churn Risk Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-box">A customer is labeled <b>churned</b> if they placed no '
        f'order in the {CHURN_RECENCY_DAYS} days before {ANALYSIS_DATE}. A Random Forest model '
        f'is trained on each customer\'s tenure, order history, return behavior and '
        f'demographics (recency is <b>excluded</b> as a feature since it defines the label) '
        f'and outputs a <b>churn probability</b> for every customer. Probabilities are '
        f'bucketed into <b>Low (&lt; 0.40)</b>, <b>Medium (0.40-0.70)</b> and '
        f'<b>High (&gt; 0.70)</b> risk tiers.</div>',
        unsafe_allow_html=True,
    )
    st.caption("Reflects the Customer Filters in the sidebar.")

    rc = filtered_df["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low Risk", f"{int(rc['Low']):,}")
    c2.metric("Medium Risk", f"{int(rc['Medium']):,}")
    c3.metric("High Risk", f"{int(rc['High']):,}")
    c4.metric("Avg Churn Probability", f"{filtered_df['churn_probability'].mean()*100:.1f}%")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.histogram(
            filtered_df, x="churn_probability", color="risk_category", nbins=30,
            color_discrete_map=RISK_COLORS, title="Churn Probability Distribution",
            labels={"churn_probability": "Churn Probability"},
        )
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            importance.head(10).sort_values("importance"),
            x="importance", y="feature", orientation="h",
            title="What Drives the Model's Churn Prediction",
            color_discrete_sequence=[PRIMARY],
            labels={"importance": "Feature Importance", "feature": ""},
        )
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">Left: distribution of predicted churn probability for the '
        'filtered customers, colored by risk tier. Right: the features the model relies on '
        'most - typically <b>tenure</b> (time since signup) and <b>days to first order</b> '
        'dominate, meaning customers who took a long time to convert (or never did) are the '
        'easiest to flag as high-risk.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">High-Risk Customers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">Customers ranked by predicted churn probability (highest '
        'first). Search by name or email, or use the sidebar filters to narrow by country, '
        'traffic source, segment or risk tier, then export the list below to drive a win-back '
        'campaign.</div>',
        unsafe_allow_html=True,
    )
    search = st.text_input("Search by name or email")
    high_risk_df = filtered_df.sort_values("churn_probability", ascending=False)
    if search:
        mask = (
            high_risk_df["customer_name"].str.contains(search, case=False, na=False)
            | high_risk_df["email"].str.contains(search, case=False, na=False)
        )
        high_risk_df = high_risk_df[mask]

    display_cols = [
        "customer_name", "email", "country", "segment_name", "rfm_segment",
        "frequency", "monetary", "recency_days", "churn_probability", "risk_category",
    ]
    st.dataframe(
        high_risk_df[display_cols].head(500),
        hide_index=True,
        use_container_width=True,
        column_config={
            "customer_name": "Customer",
            "segment_name": "Behavioral Segment",
            "rfm_segment": "RFM Segment",
            "frequency": "Orders",
            "monetary": st.column_config.NumberColumn("Lifetime Value", format="$%.2f"),
            "recency_days": "Days Since Last Order",
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Probability", min_value=0, max_value=1, format="%.2f"
            ),
            "risk_category": "Risk",
        },
    )
    st.download_button(
        "Download filtered customer list (CSV)",
        data=high_risk_df[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

# ============================== TAB 5: GEOGRAPHIC ==============================
with tab5:
    st.markdown('<div class="section-title">Geographic Distribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">These maps show where the customer base is concentrated and '
        'how revenue, customer counts and churn risk vary by location. Use the '
        '<b>metric selector</b> below to switch what each map is shaded by: for Lifetime '
        'Revenue, Customers and High-Risk Customers, darker shading marks the largest values, '
        'while for Churn Rate it marks the markets most in need of retention attention. The '
        'world map covers every country represented in the customer base; the second map '
        'breaks the United States - the largest single market - down by state.</div>',
        unsafe_allow_html=True,
    )

    geo_metrics = {
        "Total Lifetime Revenue": ("revenue", "Blues"),
        "Customers": ("customers", "Blues"),
        "Churn Rate (%)": ("churn_rate", "OrRd"),
        "High-Risk Customers": ("high_risk_customers", "OrRd"),
    }
    geo_metric_label = st.selectbox("Map metric", list(geo_metrics.keys()))
    metric_col, color_scale = geo_metrics[geo_metric_label]

    country_geo, state_geo = compute_geo_data()

    fig = px.choropleth(
        country_geo, locations="country", locationmode="country names",
        color=metric_col, color_continuous_scale=color_scale,
        hover_name="country", labels=GEO_LABELS, hover_data=GEO_HOVER,
        title=f"{geo_metric_label} by Country",
    )
    fig.update_layout(height=460, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    top_countries = country_geo.sort_values(metric_col, ascending=False).head(10)
    st.dataframe(
        top_countries[["country", "customers", "revenue", "churn_rate", "high_risk_customers"]],
        hide_index=True, use_container_width=True,
        column_config={
            "country": "Country",
            "customers": "Customers",
            "revenue": st.column_config.NumberColumn("Lifetime Revenue", format="$%.0f"),
            "churn_rate": st.column_config.NumberColumn("Churn Rate %", format="%.1f%%"),
            "high_risk_customers": "High-Risk Customers",
        },
    )

    # --- Country-level data analysis --------------------------------------
    top3 = country_geo.sort_values("revenue", ascending=False).head(3)
    top3_revenue = top3["revenue"].sum()
    top3_share = top3_revenue / country_geo["revenue"].sum() * 100
    top3_names = ", ".join(top3["country"].iloc[:-1]) + " and " + top3["country"].iloc[-1]

    major_markets = country_geo[country_geo["customers"] >= 1000]
    min_churn_row = major_markets.loc[major_markets["churn_rate"].idxmin()]
    max_churn_row = major_markets.loc[major_markets["churn_rate"].idxmax()]

    st.markdown(
        f'<div class="insight-box"><b>Data analysis:</b> {len(country_geo)} countries are '
        f'represented in the customer base, but revenue is concentrated at the top: '
        f'<b>{top3_names}</b> together generate {fmt_currency(top3_revenue)}, or '
        f'<b>{top3_share:.1f}%</b> of the {fmt_currency(country_geo["revenue"].sum())} shown '
        f'on the map above. Churn, by contrast, is far more evenly spread - among markets '
        f'with at least 1,000 customers, churn rates range from only '
        f'<b>{min_churn_row["churn_rate"]:.1f}%</b> in {min_churn_row["country"]} to '
        f'<b>{max_churn_row["churn_rate"]:.1f}%</b> in {max_churn_row["country"]}. That '
        f'narrow range suggests churn is driven by company-wide factors - such as how '
        f'quickly a new customer makes a second purchase - rather than by anything specific '
        f'to a given country, so retention programs do not need to be heavily localized to '
        f'be effective; they should instead be prioritized by market size, starting with '
        f'{top3_names}.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">U.S. States</div>', unsafe_allow_html=True)

    fig = px.choropleth(
        state_geo, locations="state_code", locationmode="USA-states", scope="usa",
        color=metric_col, color_continuous_scale=color_scale,
        hover_name="state", labels=GEO_LABELS, hover_data=GEO_HOVER,
        title=f"{geo_metric_label} by U.S. State",
    )
    fig.update_layout(height=460, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    us_row = country_geo.loc[country_geo["country"] == "United States"].iloc[0]
    us_total = int(us_row["customers"])
    top_state_n = state_geo.sort_values("customers", ascending=False).iloc[0]
    state_by_revenue = state_geo.sort_values("revenue", ascending=False)
    top_state_rev, second_state_rev = state_by_revenue.iloc[0], state_by_revenue.iloc[1]
    top_state_rev_share = top_state_rev["revenue"] / us_row["revenue"] * 100
    st.markdown(
        f'<div class="insight-box">The United States accounts for {fmt_int(us_total)} of the '
        f'{fmt_int(total_customers)} customers in the dataset '
        f'({us_total / total_customers * 100:.1f}%). <b>{top_state_n["state"]}</b> is the '
        f'largest U.S. market by customer count, with {fmt_int(top_state_n["customers"])} '
        f'customers, while <b>{top_state_rev["state"]}</b> generates the most lifetime revenue '
        f'at {fmt_currency(top_state_rev["revenue"])}. Switch the metric selector above to '
        f'<b>Churn Rate (%)</b> to spot states where a larger share of the customer base has '
        f'gone quiet, which can help prioritize regional retention campaigns.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="insight-box"><b>Data analysis:</b> {top_state_rev["state"]} alone '
        f'contributes {fmt_currency(top_state_rev["revenue"])} of the '
        f'{fmt_currency(us_row["revenue"])} generated by U.S. customers - '
        f'<b>{top_state_rev_share:.1f}%</b> of the U.S. total - with {second_state_rev["state"]} '
        f'a distant second at {fmt_currency(second_state_rev["revenue"])}. Combined with the '
        f'top-three-country concentration above, this points to a long-tail pattern that is '
        f'common in e-commerce: a small number of markets and states drive a '
        f'disproportionate share of both revenue and churn-risk exposure, so they merit '
        f'closer monitoring even though, as shown above, the underlying churn rate itself '
        f'does not vary much by location.</div>',
        unsafe_allow_html=True,
    )
