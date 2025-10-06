# Streamlit app — GrowthOracle (Module 3 only)
# Category Performance Analysis (Standalone Project)
# -------------------------------------------------
# How to run:
#   pip install streamlit pandas numpy plotly pyyaml
#   streamlit run app.py

import os, re, sys, json, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import to_html
    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    to_html = None
    _HAS_PLOTLY = False

# ---- Page ----
st.set_page_config(
    page_title="GrowthOracle — Module 3: Category Performance",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧭",
)
st.title("GrowthOracle — Module 3: Category Performance")
st.caption("Aggregate performance by category with weighted CTR/rank, per‑article efficiency, treemap & heatmap")

# ---- Logger ----
@st.cache_resource
def get_logger(level=logging.INFO):
    logger = logging.getLogger("growthoracle_mod3")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()

# ---- Defaults / Config ----
_DEFAULT_CONFIG = {
    "performance": {"sample_row_limit": 350_000, "seed": 42},
    "defaults": {"date_lookback_days": 60}
}

@st.cache_resource
def load_config():
    cfg = _DEFAULT_CONFIG.copy()
    if yaml is not None:
        for candidate in ["config.yaml", "growthoracle.yaml", "settings.yaml"]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        user_cfg = yaml.safe_load(f) or {}
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
                    logger.info(f"Loaded configuration from {candidate}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {candidate}: {e}")
    return cfg

CONFIG = load_config()

# ---- Validation Core (minimal) ----
@dataclass
class ValidationMessage:
    category: str  # "Critical" | "Warning" | "Info"
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ValidationCollector:
    def __init__(self):
        self.messages: List[ValidationMessage] = []

    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.messages:
            return pd.DataFrame(columns=["category", "code", "message", "context"])
        return pd.DataFrame([{
            "category": m.category,
            "code": m.code,
            "message": m.message,
            "context": json.dumps(m.context, ensure_ascii=False)
        } for m in self.messages])

# ---- Helpers ----
def download_df_button(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        st.warning(f"No data to download for {label}")
        return
    try:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def export_plot_html(fig, name: str):
    if to_html is None or fig is None:
        st.info("Plotly HTML export not available.")
        return
    try:
        html_str = to_html(fig, include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label=f"Export {name} (HTML)",
            data=html_str.encode("utf-8"),
            file_name=f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Failed to export plot: {e}")

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "dt", "time", "timestamp", "publish"])]

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    if col is None or len(col) == 0:
        return pd.Series([], dtype='datetime64[ns, UTC]')
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    bad = parsed.isna().sum()
    if bad > 0:
        vc.add("Warning", "DATE_PARSE", f"Unparseable datetime in {name}", bad_rows=int(bad))
    return parsed

def coerce_numeric(series, name: str, vc: ValidationCollector, clamp: Optional[Tuple[float, float]] = None) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    bad = s.isna().sum()
    if bad > 0:
        vc.add("Warning", "NUM_COERCE", f"Non-numeric values coerced to NaN in {name}", bad_rows=int(bad))
    if clamp and len(s) > 0:
        lo, hi = clamp
        s = s.clip(lower=lo, upper=hi) if hi is not None else s.clip(lower=lo)
    return s

# ---- CSV Readers ----
def read_csv_safely(upload, name: str, vc: ValidationCollector) -> Optional[pd.DataFrame]:
    if upload is None:
        vc.add("Critical", "NO_FILE", f"{name} file not provided"); return None
    try_encodings = [None, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in try_encodings:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc) if enc else pd.read_csv(upload)
            if df.empty or df.shape[1] == 0:
                vc.add("Critical", "EMPTY_CSV", f"{name} appears empty"); return None
            return df
        except Exception as e:
            last_err = e
            continue
    vc.add("Critical", "CSV_ENCODING", f"Failed to read {name}", last_error=str(last_err))
    return None

# ---- Mapping ----
def _guess_colmap(prod_df, ga4_df, gsc_df):
    if prod_df is None or gsc_df is None:
        return {}, {}, {}
    prod_map = {
        "msid": "Msid" if "Msid" in prod_df.columns else next((c for c in prod_df.columns if c.lower()=="msid"), None),
        "title": "Title" if "Title" in prod_df.columns else next((c for c in prod_df.columns if "title" in c.lower()), None),
        "path": "Path" if "Path" in prod_df.columns else next((c for c in prod_df.columns if "path" in c.lower()), None),
        "publish": "Publish Time" if "Publish Time" in prod_df.columns else next((c for c in prod_df.columns if "publish" in c.lower()), None),
    }
    ga4_map = {}
    if ga4_df is not None and not ga4_df.empty:
        ga4_map = {
            "msid": "customEvent:msid" if "customEvent:msid" in ga4_df.columns else next((c for c in ga4_df.columns if "msid" in c.lower()), None),
            "date": "date" if "date" in ga4_df.columns else next((c for c in ga4_df.columns if c.lower()=="date"), None),
            "pageviews": "screenPageViews" if "screenPageViews" in ga4_df.columns else next((c for c in ga4_df.columns if "pageview" in c.lower()), None),
            "users": "totalUsers" if "totalUsers" in ga4_df.columns else next((c for c in ga4_df.columns if "users" in c.lower()), None),
            "engagement": "userEngagementDuration" if "userEngagementDuration" in ga4_df.columns else next((c for c in ga4_df.columns if "engagement" in c.lower()), None),
            "bounce": "bounceRate" if "bounceRate" in ga4_df.columns else next((c for c in ga4_df.columns if "bounce" in c.lower()), None),
        }
    gsc_map = {
        "date": "Date" if "Date" in gsc_df.columns else next((c for c in gsc_df.columns if c.lower()=="date"), None),
        "page": "Page" if "Page" in gsc_df.columns else next((c for c in gsc_df.columns if "page" in c.lower()), None),
        "query": "Query" if "Query" in gsc_df.columns else next((c for c in gsc_df.columns if "query" in c.lower()), None),
        "clicks": "Clicks" if "Clicks" in gsc_df.columns else next((c for c in gsc_df.columns if "clicks" in c.lower()), None),
        "impr": "Impressions" if "Impressions" in gsc_df.columns else next((c for c in gsc_df.columns if "impr" in c.lower()), None),
        "ctr": "CTR" if "CTR" in gsc_df.columns else next((c for c in gsc_df.columns if "ctr" in c.lower()), None),
        "pos": "Position" if "Position" in gsc_df.columns else next((c for c in gsc_df.columns if "position" in c.lower()), None),
    }
    return prod_map, ga4_map, gsc_map

# ---- Standardization & Merge ----
def standardize_dates_early(prod_df, ga4_df, gsc_df, mappings, vc: ValidationCollector):
    p = prod_df.copy() if prod_df is not None else None
    if p is not None and mappings["prod"].get("publish") and mappings["prod"]["publish"] in p.columns:
        p["Publish Time"] = safe_dt_parse(p[mappings["prod"]["publish"]], "Publish Time", vc)

    g4 = ga4_df.copy() if ga4_df is not None else None
    if g4 is not None and mappings["ga4"].get("date") and mappings["ga4"]["date"] in g4.columns:
        g4["date"] = pd.to_datetime(g4[mappings["ga4"]["date"]], errors="coerce").dt.date

    gs = gsc_df.copy() if gsc_df is not None else None
    if gs is not None and mappings["gsc"].get("date") and mappings["gsc"]["date"] in gs.columns:
        gs["date"] = pd.to_datetime(gs[mappings["gsc"]["date"]], errors="coerce").dt.date

    return p, g4, gs


def process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    vc = ValidationCollector()
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # Rename to standard names
    std_names = {
        "prod": {"msid": "msid", "title": "Title", "path": "Path", "publish": "Publish Time"},
        "ga4": {"msid": "msid", "date": "date", "pageviews": "screenPageViews", "users": "totalUsers", "engagement": "userEngagementDuration", "bounce": "bounceRate"},
        "gsc": {"date": "date", "page": "page_url", "query": "Query", "clicks": "Clicks", "impr": "Impressions", "ctr": "CTR", "pos": "Position"}
    }
    try:
        if prod_df is not None: prod_df.rename(columns={prod_map[k]: v for k, v in std_names["prod"].items() if prod_map.get(k)}, inplace=True)
        if ga4_df is not None: ga4_df.rename(columns={ga4_map[k]: v for k, v in std_names["ga4"].items() if ga4_map.get(k)}, inplace=True)
        if gsc_df is not None: gsc_df.rename(columns={gsc_map[k]: v for k, v in std_names["gsc"].items() if gsc_map.get(k)}, inplace=True)
    except Exception as e:
        vc.add("Critical", "RENAME_FAIL", f"Column renaming failed: {e}")
        return None, vc

    # Dates
    prod_df, ga4_df, gsc_df = standardize_dates_early(prod_df, ga4_df, gsc_df, {"prod": std_names["prod"], "ga4": std_names["ga4"], "gsc": std_names["gsc"]}, vc)

    # MSID cleanup
    for df, name in [(prod_df, "Production"), (ga4_df, "GA4")]:
        if df is not None and "msid" in df.columns:
            df["msid"] = pd.to_numeric(df["msid"], errors="coerce")
            df.dropna(subset=["msid"], inplace=True)
            if not df.empty: df["msid"] = df["msid"].astype("int64")

    if gsc_df is not None and "page_url" in gsc_df.columns:
        gsc_df["msid"] = gsc_df["page_url"].astype(str).str.extract(r'(\d+)\.cms').iloc[:, 0]
        gsc_df["msid"] = pd.to_numeric(gsc_df["msid"], errors="coerce")
        gsc_df.dropna(subset=["msid"], inplace=True)
        if not gsc_df.empty: gsc_df["msid"] = gsc_df["msid"].astype("int64")

        # Numeric conversions & clamps
        for col, clamp in [("Clicks", (0, None)), ("Impressions", (0, None)), ("Position", (1, 100))]:
            if col in gsc_df.columns:
                gsc_df[col] = coerce_numeric(gsc_df[col], f"GSC.{col}", vc, clamp=clamp)

        # CTR cleanup: accept %, decimals, or compute
        if "CTR" in gsc_df.columns:
            if gsc_df["CTR"].dtype == "object":
                tmp = gsc_df["CTR"].astype(str).str.replace("%", "", regex=False).str.replace(",", "").str.strip()
                gsc_df["CTR"] = pd.to_numeric(tmp, errors="coerce") / 100.0
            gsc_df["CTR"] = coerce_numeric(gsc_df["CTR"], "GSC.CTR", vc, clamp=(0, 1))
        elif {"Clicks","Impressions"}.issubset(gsc_df.columns):
            gsc_df["CTR"] = (gsc_df["Clicks"] / gsc_df["Impressions"].replace(0, np.nan)).fillna(0)

    # Merge GSC × Prod (GA4 optional)
    if gsc_df is None or prod_df is None or gsc_df.empty or prod_df.empty:
        vc.add("Critical", "MERGE_PREP_FAIL", "Missing GSC or Production data"); return None, vc

    prod_cols = [c for c in ["msid","Title","Path","Publish Time"] if c in prod_df.columns]
    master = pd.merge(gsc_df, prod_df[prod_cols].drop_duplicates(subset=["msid"]), on="msid", how="left")

    # Enrich with categories
    if "Path" in master.columns:
        cats = master["Path"].astype(str).str.strip('/').str.split('/', n=2, expand=True)
        master["L1_Category"] = cats[0].fillna("Uncategorized")
        master["L2_Category"] = cats[1].fillna("General")
    else:
        master["L1_Category"] = "Uncategorized"
        master["L2_Category"] = "General"

    # Attach GA4 daily metrics if present
    if ga4_df is not None and not ga4_df.empty and "date" in ga4_df.columns:
        numeric_cols = [c for c in ["screenPageViews","totalUsers","userEngagementDuration","bounceRate"] if c in ga4_df.columns]
        ga4_daily = ga4_df.groupby(["msid","date"], as_index=False)[numeric_cols].sum(min_count=1)
        master = pd.merge(master, ga4_daily, on=["msid","date"], how="left")

    master["_lineage"] = "GSC→PROD→(GA4)"
    return master, vc

# ---- Category Aggregation ----
def analyze_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate category-level performance + weighted metrics."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    # Ensure categories exist
    for c in ["L1_Category", "L2_Category"]:
        if c not in d.columns:
            d[c] = "Uncategorized"

    # Numericize
    for c in ["Clicks","Impressions","screenPageViews","totalUsers","userEngagementDuration","Position","CTR"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Fallback CTR if missing
    if "CTR" not in d.columns and {"Clicks","Impressions"}.issubset(d.columns):
        d["CTR"] = (d["Clicks"] / d["Impressions"].replace(0, np.nan)).fillna(0)

    # Weight preference
    w = "Impressions" if "Impressions" in d.columns else ("Clicks" if "Clicks" in d.columns else None)
    if w is None:
        d["_w"] = 1.0
    else:
        d["_w"] = d[w].fillna(0)

    # Group
    agg = d.groupby(["L1_Category","L2_Category"], as_index=False).agg(
        total_articles=("msid","nunique"),
        total_gsc_clicks=("Clicks","sum") if "Clicks" in d.columns else ("msid","count"),
        total_impressions=("Impressions","sum") if "Impressions" in d.columns else ("msid","count"),
        total_pageviews=("screenPageViews","sum") if "screenPageViews" in d.columns else ("msid","count"),
        total_users=("totalUsers","sum") if "totalUsers" in d.columns else ("msid","count"),
        avg_engagement_s=("userEngagementDuration","mean") if "userEngagementDuration" in d.columns else ("msid","count"),
    )

    # Weighted CTR (ΣClicks / ΣImpressions)
    if {"total_gsc_clicks","total_impressions"}.issubset(agg.columns):
        agg["ctr_weighted"] = np.where(
            agg["total_impressions"] > 0,
            agg["total_gsc_clicks"] / agg["total_impressions"],
            np.nan
        )
    else:
        agg["ctr_weighted"] = np.nan

    # Weighted Position (by Impressions→best, else Clicks→else equal)
    if "Position" in d.columns:
        wp = d.groupby(["L1_Category","L2_Category"]).apply(
            lambda g: np.average(g["Position"].dropna(), weights=g["_w"].loc[g["Position"].dropna().index])
                      if g["Position"].notna().any() and g["_w"].sum() > 0 else np.nan
        ).reset_index(name="avg_position_weighted")
        agg = agg.merge(wp, on=["L1_Category","L2_Category"], how="left")
    else:
        agg["avg_position_weighted"] = np.nan

    # Per-article efficiency
    agg["users_per_article"] = np.where(agg["total_articles"]>0, agg["total_users"]/agg["total_articles"], np.nan)
    agg["pvs_per_article"]   = np.where(agg["total_articles"]>0, agg["total_pageviews"]/agg["total_articles"], np.nan)
    agg["clicks_per_article"] = np.where(agg["total_articles"]>0, agg["total_gsc_clicks"]/agg["total_articles"], np.nan)
    agg["impr_per_article"]   = np.where(agg["total_articles"]>0, agg["total_impressions"]/agg["total_articles"], np.nan)

    return agg

# Metric resolver
from typing import Tuple as _Tuple

def _resolve_cat_metric(metric_choice: str, per_article: bool) -> _Tuple[str, str, bool]:
    """
    Returns (pretty_label, df_column_name, ascending_sort)
    ascending_sort=True for 'Avg Position' (lower is better), else False.
    """
    if per_article and metric_choice in {"Users","Page Views","Clicks","Impressions"}:
        col = {
            "Users":"users_per_article",
            "Page Views":"pvs_per_article",
            "Clicks":"clicks_per_article",
            "Impressions":"impr_per_article",
        }[metric_choice]
        return f"{metric_choice} per Article", col, False

    mapping = {
        "Users": ("total_users", False),
        "Page Views": ("total_pageviews", False),
        "Clicks": ("total_gsc_clicks", False),
        "Impressions": ("total_impressions", False),
        "Avg Position": ("avg_position_weighted", True),  # lower is better
    }
    pretty = metric_choice
    col, asc = mapping.get(metric_choice, ("total_gsc_clicks", False))
    return pretty, col, asc

# Visuals

def category_treemap(cat_df: pd.DataFrame, metric_choice: str, per_article: bool):
    if not _HAS_PLOTLY:
        st.info("Treemap requires Plotly.")
        return
    if cat_df is None or cat_df.empty:
        st.info("No category data.")
        return

    pretty, col, asc = _resolve_cat_metric(metric_choice, per_article)

    df = cat_df.copy()
    # Area value
    if metric_choice == "Avg Position":
        # Area by volume (impressions) when viewing a ratio-like metric
        if "total_impressions" in df.columns:
            df["metric_value"] = pd.to_numeric(df["total_impressions"], errors="coerce").fillna(0)
        else:
            df["metric_value"] = 1
        color_col = "avg_position_weighted"
        color_title = "Avg Position (lower better)"
        color_scale = "RdYlGn_r"  # reversed (green = low number)
    else:
        df["metric_value"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        color_col = "ctr_weighted" if "ctr_weighted" in df.columns else "metric_value"
        color_title = "Weighted CTR"
        color_scale = "RdYlGn"

    fig = px.treemap(
        df,
        path=["L1_Category","L2_Category"],
        values="metric_value",
        color=color_col,
        color_continuous_scale=color_scale,
        hover_data={
            "total_articles": True,
            "total_gsc_clicks": True,
            "total_impressions": True,
            "total_pageviews": True,
            "total_users": True,
            "avg_position_weighted": ":.2f",
            "ctr_weighted": ":.2%"
        },
        title=f"Treemap — {pretty} (color = {color_title})"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    export_plot_html(fig, f"treemap_{col}")

    download_df_button(
        df[["L1_Category","L2_Category","total_articles","total_gsc_clicks","total_impressions",
            "total_pageviews","total_users","avg_position_weighted","ctr_weighted"]],
        f"treemap_data_{col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download treemap data (CSV)"
    )


def category_heatmap(cat_df: pd.DataFrame, metric_choice: str, per_article: bool):
    if not _HAS_PLOTLY:
        st.info("Heatmap requires Plotly.")
        return
    if cat_df is None or cat_df.empty:
        st.info("No category data.")
        return

    pretty, col, asc = _resolve_cat_metric(metric_choice, per_article)
    df = cat_df.copy()

    # Pivot: rows=L2, cols=L1 (keeps labels readable)
    try:
        pv = df.pivot_table(
            index="L2_Category", columns="L1_Category",
            values=col, aggfunc="sum"
        ).fillna(0)
    except Exception:
        st.info("Not enough category variety for a heatmap.")
        return

    color_scale = "RdYlGn_r" if metric_choice == "Avg Position" else "RdYlGn"
    fig = px.imshow(
        pv,
        labels=dict(x="L1 Category", y="L2 Category", color=pretty),
        color_continuous_scale=color_scale,
        title=f"Heatmap — {pretty}"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    export_plot_html(fig, f"heatmap_{col}")

    pv_reset = pv.reset_index()
    download_df_button(
        pv_reset,
        f"heatmap_matrix_{col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download heatmap matrix (CSV)"
    )

# ---- Sidebar: Filters ----
with st.sidebar:
    st.subheader("Category View Settings")
    metric_choice = st.selectbox(
        "Metric",
        ["Users","Clicks","Page Views","Impressions","Avg Position"],
        index=1
    )
    per_article = st.checkbox("Per Article", value=False, help="Efficiency view (disabled for Avg Position)")
    if metric_choice == "Avg Position":
        per_article = False
    topn = st.slider("Top N for ranking", 5, 30, 10, step=1)

    st.markdown("---")
    st.subheader("Analysis Period")
    end = date.today()
    start = end - timedelta(days=CONFIG["defaults"]["date_lookback_days"])
    start_date = st.date_input("Start Date", value=start)
    end_date = st.date_input("End Date", value=end)
    if start_date > end_date:
        st.warning("Start date is after end date. Swapping.")
        start_date, end_date = end_date, start_date

# ---- Stepper ----
st.markdown("### Onboarding & Data Ingestion")
step = st.radio("Steps", [
    "1) Get CSV Templates",
    "2) Upload & Map Columns",
    "3) Validate & Process",
    "4) Analyze (Module 3)"
], horizontal=True)

# Templates

def _make_template_production():
    return pd.DataFrame({
        "Msid": [101, 102, 103],
        "Title": ["Budget 2025 highlights explained", "IPL 2025 schedule & squads", "Monsoon updates: city-by-city guide"],
        "Path": ["/business/budget-2025/highlights", "/sports/cricket/ipl-2025/schedule", "/news/monsoon/guide"],
        "Publish Time": ["2025-08-01 09:15:00", "2025-08-10 18:30:00", "2025-09-01 07:00:00"]
    })

def _make_template_ga4():
    return pd.DataFrame({
        "customEvent:msid": [101, 101, 102, 102, 103],
        "date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "screenPageViews": [5000, 6000, 15000, 12000, 7000],
        "totalUsers": [4000, 4500, 10000, 8000, 5200],
        "userEngagementDuration": [52.3, 48.2, 41.0, 44.7, 63.1],
        "bounceRate": [0.42, 0.45, 0.51, 0.49, 0.38]
    })

def _make_template_gsc():
    return pd.DataFrame({
        "Date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "Page": [
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/news/monsoon/guide/103.cms"
        ],
        "Query": ["budget 2025", "budget highlights", "ipl 2025 schedule", "ipl squads", "monsoon city guide"],
        "Clicks": [200, 240, 1200, 1100, 300],
        "Impressions": [5000, 5500, 40000, 38000, 7000],
        "CTR": [0.04, 0.0436, 0.03, 0.0289, 0.04286],
        "Position": [8.2, 8.0, 12.3, 11.7, 9.1]
    })

if step == "1) Get CSV Templates":
    st.info("Download sample CSV templates to understand required structure.")
    colA, colB, colC = st.columns(3)
    with colA:
        df = _make_template_production(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_production.csv", "Download Production Template")
    with colB:
        df = _make_template_ga4(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_ga4.csv", "Download GA4 Template")
    with colC:
        df = _make_template_gsc(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_gsc.csv", "Download GSC Template")
    st.stop()

# Step 2: uploads + mapping
st.subheader("Upload Your Data Files")
col1, col2, col3 = st.columns(3)
with col1:
    prod_file = st.file_uploader("Production Data (CSV)", type=["csv"], key="prod_csv")
    if prod_file: st.success(f"✓ Production: {prod_file.name}")
with col2:
    ga4_file = st.file_uploader("GA4 Data (CSV) — optional", type=["csv"], key="ga4_csv")
    if ga4_file: st.success(f"✓ GA4: {ga4_file.name}")
with col3:
    gsc_file = st.file_uploader("GSC Data (CSV)", type=["csv"], key="gsc_csv")
    if gsc_file: st.success(f"✓ GSC: {gsc_file.name}")

if not all([prod_file, gsc_file]):
    st.warning("Please upload Production & GSC files to proceed"); st.stop()

vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read) if ga4_file else None
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)

if any(df is None or df.empty for df in [prod_df_raw, gsc_df_raw]):
    st.error("One or more uploaded files appear empty/unreadable.")
    st.dataframe(vc_read.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Column mapping UI
st.subheader("Column Mapping")
prod_guess, ga4_guess, gsc_guess = _guess_colmap(prod_df_raw, ga4_df_raw if ga4_df_raw is not None else pd.DataFrame(), gsc_df_raw)

with st.expander("Production Mapping", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    prod_map = {}
    prod_map["msid"] = c1.selectbox("MSID", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("msid")) if prod_guess.get("msid") in prod_df_raw.columns else 0)
    prod_map["title"] = c2.selectbox("Title", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("title")) if prod_guess.get("title") in prod_df_raw.columns else 0)
    prod_map["path"] = c3.selectbox("Path", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("path")) if prod_guess.get("path") in prod_df_raw.columns else 0)
    prod_map["publish"] = c4.selectbox("Publish Time", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("publish")) if prod_guess.get("publish") in prod_df_raw.columns else 0)

with st.expander("GA4 Mapping (optional)", expanded=False):
    if ga4_df_raw is not None:
        c1, c2, c3, c4 = st.columns(4)
        ga4_map = {}
        ga4_map["msid"] = c1.selectbox("MSID (GA4)", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("msid")) if ga4_guess.get("msid") in ga4_df_raw.columns else 0)
        ga4_map["date"] = c2.selectbox("Date (GA4)", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("date")) if ga4_guess.get("date") in ga4_df_raw.columns else 0)
        ga4_map["pageviews"] = c3.selectbox("Pageviews", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("pageviews")) if ga4_guess.get("pageviews") in ga4_df_raw.columns else 0)
        ga4_map["users"] = c4.selectbox("Users", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("users")) if ga4_guess.get("users") in ga4_df_raw.columns else 0)
    else:
        ga4_map = {}
        st.info("GA4 optional — adds users/pageviews to the aggregates when available.")

with st.expander("GSC Mapping", expanded=True):
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    gsc_map = {}
    gsc_map["date"] = c1.selectbox("Date (GSC)", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("date")) if gsc_guess.get("date") in gsc_df_raw.columns else 0)
    gsc_map["page"] = c2.selectbox("Page URL", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("page")) if gsc_guess.get("page") in gsc_df_raw.columns else 0)
    gsc_map["query"] = c3.selectbox("Query", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("query")) if gsc_guess.get("query") in gsc_df_raw.columns else 0)
    gsc_map["clicks"] = c4.selectbox("Clicks", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("clicks")) if gsc_guess.get("clicks") in gsc_df_raw.columns else 0)
    gsc_map["impr"] = c5.selectbox("Impressions", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("impr")) if gsc_guess.get("impr") in gsc_df_raw.columns else 0)
    gsc_map["ctr"] = c6.selectbox("CTR", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("ctr")) if gsc_guess.get("ctr") in gsc_df_raw.columns else 0)
    gsc_map["pos"] = c7.selectbox("Position", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("pos")) if gsc_guess.get("pos") in gsc_df_raw.columns else 0)

# Process & merge
with st.spinner("Processing & merging datasets..."):
    master_df, vc_after = process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)

if master_df is None or master_df.empty:
    st.error("Data processing failed critically. Please check mappings and file contents.")
    st.dataframe(vc_after.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Date filter
if "date" in master_df.columns:
    m = master_df.copy()
    try:
        m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.date
        mask = (m["date"] >= start_date) & (m["date"] <= end_date)
        filtered_df = m[mask].copy()
        st.info(f"Date filter applied: {len(filtered_df):,} rows from {start_date} to {end_date}")
    except Exception:
        filtered_df = master_df
else:
    filtered_df = master_df

st.success(f"✅ Master dataset ready: {filtered_df.shape[0]:,} rows × {filtered_df.shape[1]} columns")

if step != "4) Analyze (Module 3)":
    st.info("Move to **Step 4** to run the Category Performance analysis.")
    st.stop()

# -----------------------------
# ANALYSIS — Module 3 outputs
# -----------------------------
st.header("🧭 Module 3: Category Performance")

category_results = analyze_category_performance(filtered_df)
if category_results is None or category_results.empty:
    st.info("Category analysis could not be completed. Check data and mappings.")
    st.stop()

pretty, perf_col, sort_asc = _resolve_cat_metric(metric_choice, per_article)

# Show the aggregate table (optional)
with st.expander("Category table (full)", expanded=False):
    st.dataframe(category_results, use_container_width=True, hide_index=True)
    download_df_button(
        category_results,
        f"category_aggregate_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download category table (CSV)"
    )

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.subheader("Category Traffic Distribution")
    st.caption("Treemap (recommended)")
    category_treemap(category_results, metric_choice, per_article)
    st.caption("Heatmap")
    category_heatmap(category_results, metric_choice, per_article)

with col2:
    st.subheader("Top Categories by Performance")
    if perf_col in category_results.columns:
        top_cats = (category_results
                    .dropna(subset=[perf_col])
                    .sort_values(perf_col, ascending=sort_asc)
                    .head(topn))
        if metric_choice == "Avg Position":
            st.caption("Lower = better (avg rank).")
        else:
            st.caption("Higher = better.")

        if _HAS_PLOTLY:
            fig = px.bar(
                top_cats.sort_values(perf_col, ascending=not sort_asc),
                x=perf_col, y="L2_Category",
                color="L1_Category",
                orientation="h",
                title=f"Top {topn} by {pretty}"
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, f"top_categories_{perf_col}")
        else:
            st.bar_chart(top_cats.set_index("L2_Category")[perf_col])

        download_df_button(
            top_cats[["L1_Category","L2_Category", perf_col]],
            f"top_categories_{perf_col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "Download Top categories (CSV)"
        )
    else:
        st.info(f"No data for metric: {pretty}")

st.markdown("---")
st.caption("GrowthOracle — Module 3 (Standalone)")
