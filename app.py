from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    INPUT_SPECS,
    RAW_NUMERIC_FIELDS,
    calculate_cost_curve,
    default_customer,
    feature_importance_frame,
    holdout_evaluation,
    load_artifacts,
    load_datasets,
    predict_customer,
    score_batch,
)

st.set_page_config(
    page_title="ChurnIQ | Customer Retention Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
BG = "#F5F7FA"
INK = "#16212E"
MUTED = "#6B7785"
TEAL = "#0B8F83"
TEAL_DARK = "#086B64"
CORAL = "#D95F59"
AMBER = "#D99A36"
SLATE = "#526273"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#16212E; --muted:#6B7785; --teal:#0B8F83; --line:#E4EAF0; }
html, body, [class*="css"] { font-family:'DM Sans','Manrope',sans-serif; }
.stApp { background:#F5F7FA; color:var(--ink); }
.block-container { max-width:1440px; padding-top:1.65rem; padding-bottom:3.4rem; }
[data-testid="stSidebar"] { background:#101D2A; border-right:1px solid rgba(255,255,255,.07); }
[data-testid="stSidebar"] * { color:#E8EEF3; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#AAB8C5; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.38rem .58rem; border-radius:9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.07); }
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div { color:#fff; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.13); }
[data-testid="stMetric"] { background:#fff; border:1px solid #E4EAF0; border-radius:14px; padding:1rem 1.05rem; box-shadow:0 3px 12px rgba(27,42,58,.035); }
[data-testid="stMetricLabel"] { color:#6B7785; font-size:.78rem; font-weight:700; letter-spacing:.035em; text-transform:uppercase; }
[data-testid="stMetricValue"] { color:#16212E; font-weight:800; letter-spacing:-.04em; }
[data-testid="stMetricDelta"] { font-size:.76rem; }
[data-testid="stTabs"] button { font-weight:700; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#0B8F83; }
[data-testid="stForm"] { background:#fff; border:1px solid #E4EAF0; border-radius:16px; padding:1rem 1.1rem; }
[data-testid="stExpander"] { background:#fff; border:1px solid #E4EAF0; border-radius:12px; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:700; min-height:2.65rem; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:#0B8F83; border-color:#0B8F83; }
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background:#086B64; border-color:#086B64; }
[data-testid="stDataFrame"] { border:1px solid #E4EAF0; border-radius:12px; overflow:hidden; }
div[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#E4EAF0; }
.hero { position:relative; overflow:hidden; padding:1.65rem 1.85rem; border-radius:20px; color:white; background:linear-gradient(115deg,#102332 0%,#143B47 57%,#0B766F 100%); margin-bottom:1.05rem; box-shadow:0 12px 30px rgba(16,35,50,.13); }
.hero:after { content:''; position:absolute; width:320px; height:320px; border-radius:50%; right:-75px; top:-160px; border:1px solid rgba(255,255,255,.18); box-shadow:0 0 0 32px rgba(255,255,255,.025),0 0 0 72px rgba(255,255,255,.02); }
.hero-kicker { color:#8DE0D3; text-transform:uppercase; font-size:.69rem; font-weight:800; letter-spacing:.16em; margin-bottom:.55rem; }
.hero h1 { color:#fff; font-family:'Manrope',sans-serif; font-size:2rem; line-height:1.14; letter-spacing:-.045em; margin:0 0 .55rem 0; font-weight:800; }
.hero p { color:#D5E6E8; font-size:.94rem; line-height:1.55; margin:0; max-width:720px; }
.hero-meta { position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.15rem; }
.hero-chip { display:inline-flex; align-items:center; border:1px solid rgba(255,255,255,.21); color:#EAF5F4; background:rgba(255,255,255,.07); border-radius:999px; padding:.36rem .7rem; font-size:.72rem; font-weight:700; }
.page-heading { margin:0 0 .15rem 0; color:#16212E; font-size:1.65rem; line-height:1.25; font-weight:800; letter-spacing:-.04em; }
.page-subtitle { color:#6B7785; margin:0 0 1.25rem 0; font-size:.92rem; }
.section-eyebrow { color:#0B8F83; font-size:.68rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin:1.1rem 0 .35rem 0; }
.section-title { color:#16212E; font-size:1.08rem; font-weight:800; letter-spacing:-.02em; margin:0 0 .75rem 0; }
.panel { background:#fff; border:1px solid #E4EAF0; border-radius:15px; padding:1rem 1.1rem; }
.panel-title { color:#16212E; font-size:.88rem; font-weight:800; margin-bottom:.25rem; }
.panel-copy { color:#6B7785; font-size:.78rem; line-height:1.5; }
.kpi-note { color:#6B7785; font-size:.74rem; margin-top:.25rem; }
.status-pill { display:inline-block; border-radius:999px; padding:.3rem .65rem; font-weight:800; font-size:.7rem; letter-spacing:.02em; }
.status-low { color:#08776E; background:#E4F5F1; border:1px solid #BFE8DF; }
.status-mid { color:#98620D; background:#FFF4DC; border:1px solid #F4DCA7; }
.status-high { color:#A83E39; background:#FCEAE8; border:1px solid #F2C7C3; }
.result-card { background:#fff; border:1px solid #E4EAF0; border-radius:16px; padding:1.2rem 1.25rem; }
.result-number { font-size:2.4rem; line-height:1.1; letter-spacing:-.05em; font-weight:800; color:#16212E; }
.result-label { color:#6B7785; text-transform:uppercase; font-size:.69rem; letter-spacing:.11em; font-weight:800; }
.signal { padding:.68rem .8rem; background:#F6F8FA; border:1px solid #E8EDF2; border-radius:10px; margin:.42rem 0; }
.signal b { color:#253444; font-size:.82rem; }
.signal span { color:#71808E; font-size:.75rem; display:block; margin-top:.15rem; line-height:1.4; }
.notice { background:#FFF8E9; border:1px solid #F1D9A8; border-left:4px solid #D99A36; color:#73531B; border-radius:10px; padding:.8rem .95rem; font-size:.78rem; line-height:1.55; }
.footer { color:#8793A0; font-size:.7rem; border-top:1px solid #E4EAF0; padding-top:.8rem; margin-top:2.2rem; }
.small-muted { color:#7B8794; font-size:.73rem; }
@media (max-width: 700px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:1.3rem; } .hero h1 { font-size:1.6rem; } .hero:after { opacity:.4; } }
</style>
"""


@st.cache_data(show_spinner="Loading customer data…")
def get_data():
    return load_datasets()


@st.cache_resource(show_spinner="Loading LightGBM model…")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Rebuilding the original holdout evaluation…")
def get_evaluation(_model, _scaler, model_df):
    return holdout_evaluation(_model, _scaler, model_df)


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def section_heading(title: str, subtitle: str | None = None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="small-muted" style="margin-top:-.55rem;margin-bottom:.75rem">{subtitle}</div>', unsafe_allow_html=True)


def page_heading(title: str, subtitle: str):
    st.markdown(f'<div class="page-heading">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def plot_style(fig, height: int = 320):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=45, b=12),
        font=dict(family="DM Sans, sans-serif", color="#526273", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#ECF0F4", zeroline=False)
    return fig


def make_hero(model_df: pd.DataFrame, model):
    feature_count = getattr(model, "n_features_in_", "—")
    customers = len(model_df)
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">E-commerce · Predictive retention</div>
          <h1>Customer Churn<br/>Intelligence</h1>
          <p>Turn customer behaviour into clearer retention decisions. Explore churn patterns,
          score individual profiles and tune outreach using your trained LightGBM model.</p>
          <div class="hero-meta">
            <span class="hero-chip">◈ LightGBM classifier</span>
            <span class="hero-chip">{feature_count} model features</span>
            <span class="hero-chip">{customers:,} customer records</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(raw_df: pd.DataFrame, model_df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:.3rem .05rem 1.1rem .05rem">
              <div style="font-size:.68rem;letter-spacing:.18em;color:#71D5C6;font-weight:800">RETENTION SYSTEMS</div>
              <div style="font-size:1.45rem;font-weight:800;letter-spacing:-.06em;color:#FFFFFF;margin-top:.15rem">CHURN<span style="color:#71D5C6">IQ</span></div>
              <div style="font-size:.72rem;color:#94A8B8;margin-top:.1rem">Customer intelligence workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='font-size:.65rem;letter-spacing:.14em;color:#8398A8;font-weight:800;margin:.35rem 0 .4rem'>WORKSPACE</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            ["Command centre", "Customer scoring", "Segments", "Threshold lab", "Data explorer"],
            label_visibility="collapsed",
            format_func=lambda x: {
                "Command centre": "◫   Command centre",
                "Customer scoring": "◎   Customer scoring",
                "Segments": "◉   Segments",
                "Threshold lab": "⌁   Threshold lab",
                "Data explorer": "▦   Data explorer",
            }[x],
        )
        st.markdown("---")
        st.markdown("<div style='font-size:.65rem;letter-spacing:.14em;color:#8398A8;font-weight:800;margin-bottom:.55rem'>DECISION SETTINGS</div>", unsafe_allow_html=True)
        threshold = st.slider(
            "Retention decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="A customer is flagged for retention outreach when the predicted churn probability is at or above this value.",
        )
        st.caption(f"Profiles at or above **{threshold:.0%}** are flagged.")
        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size:.68rem;color:#8FA3B2;line-height:1.65">
              <div style="color:#8DE0D3;font-weight:800;letter-spacing:.12em;margin-bottom:.28rem">SYSTEM STATUS</div>
              <div>● Model artifacts loaded</div>
              <div>● Source data · {len(raw_df):,} records</div>
              <div>● Model dataset · {len(model_df):,} records</div>
              <div style="margin-top:.7rem;color:#71889A">Built with Streamlit + LightGBM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page, threshold


def render_validation_notice():
    st.markdown(
        """
        <div class="notice"><b>Model validation note.</b> The supplied notebook fits its IsolationForest anomaly detector
        before the train/test split and includes <code>Churned</code> in the numeric columns used to fit it.
        That creates a target-leakage risk in the reported validation metrics. The training set is also SMOTE-balanced,
        so <code>predict_proba</code> outputs should be checked/calibrated against the natural customer churn rate.
        The app reproduces the saved model's holdout results for traceability, but those metrics are not production-grade
        evidence until these issues are corrected and the model is retrained.</div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(raw_df: pd.DataFrame, model_df: pd.DataFrame, model, evaluation: dict):
    make_hero(model_df, model)
    churn = pd.to_numeric(raw_df["Churned"], errors="coerce")
    churn_count = int(churn.sum())
    churn_rate = float(churn.mean())
    retained_count = int((churn == 0).sum())
    ltv_mean = float(pd.to_numeric(raw_df["Lifetime_Value"], errors="coerce").mean())
    revenue_at_risk = float(
        pd.to_numeric(raw_df.loc[raw_df["Churned"] == 1, "Lifetime_Value"], errors="coerce").sum()
    )

    cols = st.columns(4)
    cols[0].metric("Customers analysed", f"{len(raw_df):,}", "Source dataset")
    cols[1].metric("Observed churn rate", pct(churn_rate), f"{churn_count:,} churned")
    cols[2].metric("Average lifetime value", money(ltv_mean), "Across all customers")
    cols[3].metric("Churned customer LTV", money(revenue_at_risk), "Historical LTV sum")

    st.markdown('<div class="section-eyebrow">MODEL VALIDATION</div>', unsafe_allow_html=True)
    section_heading("Saved model · reconstructed holdout", "Metrics are computed on the original stratified 80/20 split from the supplied notebook.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", pct(evaluation["accuracy"]))
    m2.metric("Precision", pct(evaluation["precision"]))
    m3.metric("Recall", pct(evaluation["recall"]))
    m4.metric("ROC-AUC", f"{evaluation['roc_auc']:.3f}")
    st.caption("At the notebook's 0.50 threshold. These reproduce the notebook's reported LightGBM results; see the validation caveat below.")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        section_heading("Churn composition", "Observed outcome labels in the source cohort")
        pie = pd.DataFrame({"Status": ["Retained", "Churned"], "Customers": [retained_count, churn_count]})
        fig = px.pie(
            pie, names="Status", values="Customers", hole=0.68,
            color="Status", color_discrete_map={"Retained": TEAL, "Churned": CORAL},
        )
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#FFFFFF", width=3)))
        fig.add_annotation(text=f"{churn_rate:.1%}<br><span style='font-size:10px;color:#7B8794'>churn rate</span>", x=.5, y=.5, showarrow=False, font=dict(size=21, color=INK))
        plot_style(fig, 305)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with right:
        section_heading("Churn by country", "Countries ranked by observed churn rate")
        country = raw_df.groupby("Country", dropna=False).agg(
            customers=("Churned", "size"), churn_rate=("Churned", "mean")
        ).reset_index().sort_values("churn_rate", ascending=True)
        fig = px.bar(
            country, x="churn_rate", y="Country", orientation="h", text=country["churn_rate"].map(lambda x: f"{x:.1%}"),
            color="churn_rate", color_continuous_scale=["#BFE8DF", TEAL],
            labels={"churn_rate": "Churn rate", "Country": ""},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickformat=".0%", range=[0, min(1.0, max(.5, country.churn_rate.max() * 1.18))])
        fig.update_coloraxes(showscale=False)
        plot_style(fig, 305)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        section_heading("Signals used by the model", "Global LightGBM split importance · not a customer-level explanation")
        importance = feature_importance_frame(model, 10)
        if not importance.empty:
            importance["Label"] = importance["Feature"].str.replace("_", " ", regex=False)
            fig = px.bar(
                importance.sort_values("Importance", ascending=True), x="Importance", y="Label", orientation="h",
                color="Importance", color_continuous_scale=["#BFE8DF", TEAL],
                labels={"Label": "", "Importance": "Split importance"},
            )
            fig.update_coloraxes(showscale=False)
            plot_style(fig, 350)
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        else:
            st.info("Feature importance is not available from this model artifact.")
    with right:
        section_heading("Customer segments", "Labels exported from the notebook")
        seg = model_df.groupby("Segment_Name").agg(
            Customers=("Churned", "size"),
            Churn_Rate=("Churned", "mean"),
            Avg_LTV=("Lifetime_Value", "mean"),
        ).reset_index().sort_values("Customers", ascending=False)
        fig = px.bar(seg, x="Customers", y="Segment_Name", orientation="h", text="Customers", color="Churn_Rate", color_continuous_scale=["#BFE8DF", CORAL], labels={"Segment_Name": "", "Customers": "Customers"})
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        plot_style(fig, 260)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        if len(seg) <= 2 or (seg["Customers"].max() / seg["Customers"].sum()) > .95:
            st.warning("The exported segmentation is highly imbalanced. Validate the notebook's cluster-to-segment mapping before using these labels for campaign targeting.")

    render_validation_notice()


def make_input_control(column: str, default: float, container):
    spec = INPUT_SPECS[column]
    value = float(np.clip(default, spec["min"], spec["max"]))
    return container.number_input(
        spec["label"],
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        value=value,
        step=float(spec["step"]),
        format=spec["format"],
        key=f"customer_{column}",
    )


def profile_signals(customer: dict, raw_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Transparent, descriptive profile checks; not SHAP or causal explanations."""
    checks = []
    metrics = [
        ("Days_Since_Last_Purchase", "high", "Purchase recency", "Review reactivation timing and replenishment reminders."),
        ("Cart_Abandonment_Rate", "high", "Cart abandonment", "Review checkout friction and test an abandoned-cart follow-up."),
        ("Customer_Service_Calls", "high", "Support contact volume", "Review unresolved issues and consider a proactive service check-in."),
        ("Email_Open_Rate", "low", "Email engagement", "Consider a different message, cadence or communication channel."),
    ]
    for column, direction, title, recommendation in metrics:
        if column not in raw_df:
            continue
        series = pd.to_numeric(raw_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        boundary = float(series.quantile(.75 if direction == "high" else .25))
        value = float(customer.get(column, 0))
        if (direction == "high" and value >= boundary) or (direction == "low" and value <= boundary):
            checks.append((title, recommendation))
    if not checks:
        checks.append(("No standout rule-based signal", "The selected profile is not extreme on the simple checks shown here. Continue to use the model score alongside customer context."))
    return checks


def render_single_customer(raw_df: pd.DataFrame, model_df: pd.DataFrame, model, scaler, threshold: float):
    defaults = default_customer(raw_df, model_df)

    # Keep location controls outside the form so City options respond immediately to Country.
    country_options = sorted(raw_df["Country"].dropna().astype(str).unique().tolist())
    selected_country = st.selectbox(
        "Country", country_options,
        index=country_options.index(defaults.get("Country")) if defaults.get("Country") in country_options else 0,
        key="customer_Country",
    )
    city_options = sorted(raw_df.loc[raw_df["Country"].astype(str) == selected_country, "City"].dropna().astype(str).unique().tolist())
    if not city_options:
        city_options = sorted(raw_df["City"].dropna().astype(str).unique().tolist())
    if "customer_City" in st.session_state and st.session_state["customer_City"] not in city_options:
        st.session_state["customer_City"] = city_options[0]
    selected_city = st.selectbox(
        "City", city_options,
        index=city_options.index(st.session_state.get("customer_City", defaults.get("City"))) if st.session_state.get("customer_City", defaults.get("City")) in city_options else 0,
        key="customer_City",
    )

    with st.form("single_customer_form", clear_on_submit=False):
        st.markdown('<div class="section-eyebrow" style="margin-top:0">01 · CUSTOMER PROFILE</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        age = make_input_control("Age", defaults["Age"], a)
        membership = make_input_control("Membership_Years", defaults["Membership_Years"], b)
        gender_options = sorted(raw_df["Gender"].dropna().astype(str).unique().tolist())
        gender = c.selectbox("Gender", gender_options, index=gender_options.index(defaults["Gender"]) if defaults["Gender"] in gender_options else 0, key="customer_Gender")
        quarter = a.selectbox("Signup quarter", ["Q1", "Q2", "Q3", "Q4"], index=["Q1", "Q2", "Q3", "Q4"].index(defaults.get("Signup_Quarter", "Q1")) if defaults.get("Signup_Quarter", "Q1") in ["Q1", "Q2", "Q3", "Q4"] else 0, key="customer_Signup_Quarter")

        st.markdown('<div class="section-eyebrow">02 · DIGITAL ENGAGEMENT</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        logins = make_input_control("Login_Frequency", defaults["Login_Frequency"], a)
        session = make_input_control("Session_Duration_Avg", defaults["Session_Duration_Avg"], b)
        pages = make_input_control("Pages_Per_Session", defaults["Pages_Per_Session"], c)
        cart = make_input_control("Cart_Abandonment_Rate", defaults["Cart_Abandonment_Rate"], a)
        wishlist = make_input_control("Wishlist_Items", defaults["Wishlist_Items"], b)
        mobile = make_input_control("Mobile_App_Usage", defaults["Mobile_App_Usage"], c)
        social = make_input_control("Social_Media_Engagement_Score", defaults["Social_Media_Engagement_Score"], a)
        email = make_input_control("Email_Open_Rate", defaults["Email_Open_Rate"], b)

        st.markdown('<div class="section-eyebrow">03 · PURCHASE BEHAVIOUR</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        purchases = make_input_control("Total_Purchases", defaults["Total_Purchases"], a)
        order_value = make_input_control("Average_Order_Value", defaults["Average_Order_Value"], b)
        recency = make_input_control("Days_Since_Last_Purchase", defaults["Days_Since_Last_Purchase"], c)
        discount = make_input_control("Discount_Usage_Rate", defaults["Discount_Usage_Rate"], a)
        returns = make_input_control("Returns_Rate", defaults["Returns_Rate"], b)
        lifetime_value = make_input_control("Lifetime_Value", defaults["Lifetime_Value"], c)
        credit = make_input_control("Credit_Balance", defaults["Credit_Balance"], a)
        payment_diversity = make_input_control("Payment_Method_Diversity", defaults["Payment_Method_Diversity"], b)

        st.markdown('<div class="section-eyebrow">04 · CUSTOMER SUPPORT</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        service_calls = make_input_control("Customer_Service_Calls", defaults["Customer_Service_Calls"], a)
        reviews = make_input_control("Product_Reviews_Written", defaults["Product_Reviews_Written"], b)
        st.markdown("<div class='small-muted'>Membership stage and seasonal activity are derived automatically to mirror the supplied notebook.</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Calculate churn probability  →", type="primary", width="stretch")

    if submitted:
        customer = {
            "Age": age,
            "Membership_Years": membership,
            "Gender": gender,
            "Country": selected_country,
            "City": selected_city,
            "Signup_Quarter": quarter,
            "Login_Frequency": logins,
            "Session_Duration_Avg": session,
            "Pages_Per_Session": pages,
            "Cart_Abandonment_Rate": cart,
            "Wishlist_Items": wishlist,
            "Mobile_App_Usage": mobile,
            "Social_Media_Engagement_Score": social,
            "Email_Open_Rate": email,
            "Total_Purchases": purchases,
            "Average_Order_Value": order_value,
            "Days_Since_Last_Purchase": recency,
            "Discount_Usage_Rate": discount,
            "Returns_Rate": returns,
            "Lifetime_Value": lifetime_value,
            "Credit_Balance": credit,
            "Payment_Method_Diversity": payment_diversity,
            "Customer_Service_Calls": service_calls,
            "Product_Reviews_Written": reviews,
        }
        try:
            result = predict_customer(customer, model, scaler, raw_df, model_df, threshold=threshold, anomaly_flag=0.0)
            st.session_state["latest_customer_result"] = {"customer": customer, "result": {k: v for k, v in result.items() if k != "feature_frame"}}
        except Exception as exc:
            st.error(f"The customer profile could not be scored: {exc}")

    latest = st.session_state.get("latest_customer_result")
    if latest:
        customer = latest["customer"]
        result = latest["result"]
        st.markdown("---")
        st.markdown('<div class="section-eyebrow">MODEL OUTPUT</div>', unsafe_allow_html=True)
        section_heading("Churn risk assessment", "The score is the model's predict_proba output, not a certainty; calibration should be checked before operational use.")
        col_a, col_b = st.columns([.95, 1.05], gap="large")
        probability = result["probability"]
        with col_a:
            status = "High" if result["flagged"] else "Below threshold"
            status_class = "status-high" if result["flagged"] else "status-low"
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-label">Predicted churn probability</div>
                  <div class="result-number">{probability:.1%}</div>
                  <div style="margin-top:.65rem"><span class="status-pill {status_class}">{status}</span>
                  <span class="small-muted" style="margin-left:.45rem">Risk band: {result['risk_level']}</span></div>
                  <div style="height:8px;background:#EAF0F3;border-radius:99px;overflow:hidden;margin-top:1rem">
                    <div style="width:{probability*100:.1f}%;height:100%;background:{CORAL if result['flagged'] else TEAL};border-radius:99px"></div>
                  </div>
                  <div class="kpi-note">Decision threshold: {threshold:.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result["flagged"]:
                st.error("Flagged for retention outreach at the selected threshold.")
            else:
                st.success("Below the selected retention threshold.")
        with col_b:
            section_heading("Suggested review signals", "Simple profile checks · not model explanations")
            for title, recommendation in profile_signals(customer, raw_df)[:4]:
                st.markdown(f"<div class='signal'><b>{title}</b><span>{recommendation}</span></div>", unsafe_allow_html=True)
        st.caption("This result uses the stored anomaly feature's neutral reference value (0) for new profiles. See the model validation note on the Command Centre.")


def render_batch_scoring(raw_df: pd.DataFrame, model_df: pd.DataFrame, model, scaler, threshold: float):
    st.markdown(
        "<div class='panel'><div class='panel-title'>Expected input</div><div class='panel-copy'>Use the original raw customer columns: "
        + ", ".join(RAW_NUMERIC_FIELDS[:7])
        + " and the remaining numeric fields, plus Gender, Country, City and Signup_Quarter. Missing fields are filled with training medians/modes. Include a stable Customer_ID column when available.</div></div>",
        unsafe_allow_html=True,
    )
    template = default_customer(raw_df, model_df)
    template_frame = pd.DataFrame([template])
    st.download_button(
        "Download one-row CSV template",
        data=template_frame.to_csv(index=False).encode("utf-8"),
        file_name="churn_customer_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_customer_csv")
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(f"**{len(batch_df):,}** customer rows loaded · {len(batch_df.columns)} columns detected")
            with st.expander("Preview uploaded rows"):
                st.dataframe(batch_df.head(10), width="stretch")
            if st.button("Score uploaded customers", type="primary", key="score_batch_btn"):
                with st.spinner("Scoring customer rows…"):
                    output = score_batch(batch_df, model, scaler, raw_df, model_df, threshold=threshold)
                st.session_state["latest_batch_output"] = output
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

    output = st.session_state.get("latest_batch_output")
    if output is not None:
        st.markdown("---")
        flagged_count = int(output["Flagged_for_Retention"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers scored", f"{len(output):,}")
        c2.metric("Flagged for retention", f"{flagged_count:,}")
        c3.metric("Flag rate", pct(flagged_count / max(len(output), 1)))
        st.dataframe(
            output.sort_values("Churn_Probability", ascending=False),
            width="stretch",
            height=360,
        )
        st.download_button(
            "Download scored customer worklist",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="churn_scored_customers.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption("For new batch rows without Anomaly_Flag/anomaly_iforest, the same neutral anomaly reference used by single-customer scoring is applied.")


def render_customer_scoring(raw_df: pd.DataFrame, model_df: pd.DataFrame, model, scaler, threshold: float):
    page_heading("Customer scoring", "Build a customer profile or upload a portfolio and estimate churn probability with the saved LightGBM model.")
    tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring"])
    with tab_single:
        render_single_customer(raw_df, model_df, model, scaler, threshold)
    with tab_batch:
        render_batch_scoring(raw_df, model_df, model, scaler, threshold)


def render_segments(model_df: pd.DataFrame):
    page_heading("Customer segments", "Explore the cluster labels exported from the notebook and compare their observed profiles.")
    if "Segment_Name" not in model_df.columns:
        st.warning("Segment_Name is missing from the model-ready dataset.")
        return
    grouped = model_df.groupby("Segment_Name", dropna=False)
    summary = grouped.agg(
        Customers=("Churned", "size"),
        Churn_Rate=("Churned", "mean"),
        Avg_Lifetime_Value=("Lifetime_Value", "mean"),
        Avg_Purchases=("Total_Purchases", "mean"),
        Avg_Discount_Usage=("Discount_Usage_Rate", "mean"),
        Avg_Returns_Rate=("Returns_Rate", "mean"),
        Avg_Email_Open_Rate=("Email_Open_Rate", "mean"),
    ).reset_index().sort_values("Churn_Rate", ascending=False)
    overall = float(model_df["Churned"].mean())
    top_share = float(summary["Customers"].max() / summary["Customers"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Segments in export", f"{len(summary)}")
    c2.metric("Overall churn rate", pct(overall))
    c3.metric("Largest segment share", pct(top_share))

    if len(summary) <= 2 or top_share > .95:
        st.warning("The exported labels are highly imbalanced: almost all records fall into one segment. The notebook's cluster-to-segment mapping should be reviewed before using these names as an operational segmentation strategy.")

    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Churn rate by segment")
        fig = px.bar(summary.sort_values("Churn_Rate"), x="Churn_Rate", y="Segment_Name", orientation="h", text=summary.sort_values("Churn_Rate")["Churn_Rate"].map(lambda v: f"{v:.1%}"), color="Churn_Rate", color_continuous_scale=["#BFE8DF", CORAL], labels={"Segment_Name": "", "Churn_Rate": "Churn rate"})
        fig.add_vline(x=overall, line_dash="dash", line_color=SLATE, annotation_text=f"Overall {overall:.1%}")
        fig.update_xaxes(tickformat=".0%")
        fig.update_coloraxes(showscale=False)
        plot_style(fig, 330)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with right:
        section_heading("Customer count by segment")
        fig = px.bar(summary.sort_values("Customers"), x="Customers", y="Segment_Name", orientation="h", text="Customers", color_discrete_sequence=[TEAL], labels={"Segment_Name": "", "Customers": "Customers"})
        fig.update_traces(textposition="outside")
        plot_style(fig, 330)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    section_heading("Segment profile")
    display = summary.copy()
    display["Churn_Rate"] = display["Churn_Rate"].map(pct)
    display["Avg_Lifetime_Value"] = display["Avg_Lifetime_Value"].map(money)
    for col in ("Avg_Purchases", "Avg_Discount_Usage", "Avg_Returns_Rate", "Avg_Email_Open_Rate"):
        display[col] = display[col].map(lambda v: f"{v:.1f}")
    st.dataframe(display, hide_index=True, width="stretch")

    st.markdown("<div class='panel'><div class='panel-title'>Notebook strategy notes</div><div class='panel-copy'>The supplied notebook recommends VIP concierge and exclusive previews for High-Value Loyal; cross-sell bundles and loyalty tiers for Growth Potential; limited-time bundles and service follow-ups for At-Risk Bargain Hunters; and value education with micro-incentives for Cost-Sensitive customers. Those labels are only present where exported in the current data.</div></div>", unsafe_allow_html=True)


def render_threshold_lab(evaluation: dict, threshold: float):
    page_heading("Threshold lab", "Tune the retention decision using the costs of missed churners and unnecessary offers.")
    st.caption("The chart uses the reconstructed 10,000-row holdout split from the notebook. The target-leakage caveat still applies.")
    left, right = st.columns(2)
    cost_fn = left.number_input("Cost of a missed churner ($)", min_value=0, max_value=10000, value=500, step=50, key="threshold_cost_fn")
    cost_fp = right.number_input("Cost of an unnecessary retention offer ($)", min_value=0, max_value=10000, value=50, step=10, key="threshold_cost_fp")
    curve = calculate_cost_curve(evaluation["y_true"], evaluation["probabilities"], cost_fn, cost_fp)
    best = curve.loc[curve["expected_cost"].idxmin()]
    current = curve.iloc[(curve["threshold"] - float(threshold)).abs().argmin()]
    savings = float(current["expected_cost"] - best["expected_cost"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cost-optimal threshold", f"{best['threshold']:.2f}")
    m2.metric("Minimum expected cost", money(best["expected_cost"]))
    m3.metric("Missed churners", f"{int(best['fn']):,}")
    m4.metric("Unnecessary offers", f"{int(best['fp']):,}")

    fig = px.line(curve, x="threshold", y="expected_cost", markers=True, labels={"threshold": "Decision threshold", "expected_cost": "Expected cost ($)"})
    fig.add_vline(x=float(best["threshold"]), line_dash="dash", line_color=CORAL, annotation_text=f"Lowest cost · {best['threshold']:.2f}")
    fig.add_vline(x=float(threshold), line_dash="dot", line_color=SLATE, annotation_text=f"Current · {threshold:.2f}")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    plot_style(fig, 370)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    if savings > 0:
        st.success(f"On this holdout split, the cost-optimal threshold reduces estimated error cost by {money(savings)} versus your current {threshold:.2f} threshold. Current-threshold cost: {money(current['expected_cost'])}.")
    else:
        st.info(f"Your current {threshold:.2f} threshold is already at or below the estimated cost of the selected optimum. Current-threshold cost: {money(current['expected_cost'])}.")
    st.caption("Cost assumptions are user-defined scenario inputs, not measured business losses. Validate them with finance/retention teams before operational use.")


def render_data_explorer(raw_df: pd.DataFrame, model_df: pd.DataFrame):
    page_heading("Data explorer", "Inspect source-data quality, distributions and a sample of the model-ready records.")
    missing = raw_df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_total = int(raw_df.isna().sum().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Source columns", f"{raw_df.shape[1]}")
    c2.metric("Missing cells", f"{missing_total:,}")
    c3.metric("Model-ready columns", f"{model_df.shape[1]}")

    tab_dist, tab_missing, tab_preview = st.tabs(["Distributions", "Missing values", "Dataset preview"])
    with tab_dist:
        candidate_features = [c for c in ["Age", "Membership_Years", "Login_Frequency", "Total_Purchases", "Average_Order_Value", "Days_Since_Last_Purchase", "Lifetime_Value", "Cart_Abandonment_Rate", "Email_Open_Rate", "Returns_Rate"] if c in raw_df.columns]
        selected = st.selectbox("Choose a feature", candidate_features)
        frame = raw_df[[selected, "Churned"]].copy()
        frame["Churn status"] = frame["Churned"].map({0: "Retained", 1: "Churned"})
        fig = px.histogram(frame, x=selected, color="Churn status", barmode="overlay", opacity=.72, nbins=38, color_discrete_map={"Retained": TEAL, "Churned": CORAL}, labels={"Churn status": ""})
        plot_style(fig, 360)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with tab_missing:
        if missing.empty:
            st.success("No missing values in the source dataset.")
        else:
            missing_frame = (
                (missing / len(raw_df) * 100)
                .rename("Missing %")
                .rename_axis("Feature")
                .reset_index()
                .sort_values("Missing %", ascending=False)
            )
            fig = px.bar(missing_frame, x="Missing %", y="Feature", orientation="h", color="Missing %", color_continuous_scale=["#F6DDAA", AMBER], labels={"Feature": ""})
            fig.update_coloraxes(showscale=False)
            plot_style(fig, max(300, 25 * len(missing_frame) + 100))
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
            st.caption("The notebook applies KNN imputation to selected engagement fields, iterative imputation to selected monetary/purchase fields, domain defaults to categorical fields and medians to the remaining numeric fields.")
    with tab_preview:
        st.markdown("**Raw source data**")
        st.dataframe(raw_df.head(15), width="stretch", height=300)
        st.markdown("**Model-ready data**")
        st.dataframe(model_df.head(15), width="stretch", height=300)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    try:
        raw_df, model_df = get_data()
        model, scaler, features = get_artifacts()
        evaluation = get_evaluation(model, scaler, model_df)
    except Exception as exc:
        st.error(f"The application could not initialise: {exc}")
        st.info("Confirm that artifacts/ contains the three supplied pickle files and data/ contains both CSV datasets. See README.md for the expected project layout.")
        st.stop()

    page, threshold = render_sidebar(raw_df, model_df)
    if page == "Command centre":
        render_overview(raw_df, model_df, model, evaluation)
    elif page == "Customer scoring":
        render_customer_scoring(raw_df, model_df, model, scaler, threshold)
    elif page == "Segments":
        render_segments(model_df)
    elif page == "Threshold lab":
        render_threshold_lab(evaluation, threshold)
    elif page == "Data explorer":
        render_data_explorer(raw_df, model_df)

    st.markdown(
        "<div class='footer'>CHURNIQ · Customer Retention Intelligence &nbsp;·&nbsp; Research prototype, not an autonomous decision system. Review model limits before using predictions for customer treatment.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
