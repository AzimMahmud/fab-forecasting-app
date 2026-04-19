"""
================================================================================
  FABRIC CONSUMPTION FORECASTING SYSTEM — PRODUCTION WEB APPLICATION
  Version  : 3.0.0
  Developer: Azim Mahmud
  Date     : April 2026
  Unit     : yards
================================================================================
"""

import json
import logging
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import joblib

    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import tensorflow as tf

    TF_OK = True
except ImportError:
    TF_OK = False

st.set_page_config(
    page_title="Fabric Forecast Pro",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================
METERS_TO_YARDS = 1.0936132983
BASE_CONSUMPTION_M = {
    "T-Shirt": 1.20,
    "Shirt": 1.80,
    "Pants": 2.50,
    "Dress": 3.00,
    "Jacket": 3.50,
}
FABRIC_COST_PER_M = {
    "Cotton": 8.50,
    "Polyester": 6.20,
    "Cotton-Blend": 7.00,
    "Silk": 25.00,
    "Denim": 9.50,
}
COMPLEXITY_MULT = {"Simple": 1.00, "Medium": 1.15, "Complex": 1.35}
BOM_BUFFER = 1.05
GARMENT_BASE_YD = {
    k: round(v * METERS_TO_YARDS, 6) for k, v in BASE_CONSUMPTION_M.items()
}
FABRIC_COST_YD = {
    k: round(v / METERS_TO_YARDS, 4) for k, v in FABRIC_COST_PER_M.items()
}

# ============================================================================
# UI CONSTANTS
# ============================================================================
GARMENT_TYPES = ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"]
FABRIC_TYPES = ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"]
FABRIC_WIDTHS_INCHES = [55, 59, 63, 71]
PATTERN_COMPLEXITIES = ["Simple", "Medium", "Complex"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]
GARMENT_ENC = {"Dress": 0, "Jacket": 1, "Pants": 2, "Shirt": 3, "T-Shirt": 4}
FABRIC_ENC = {"Cotton": 0, "Cotton-Blend": 1, "Denim": 2, "Polyester": 3, "Silk": 4}
COMPLEXITY_ENC = {"Complex": 0, "Medium": 1, "Simple": 2}
SEASON_ENC = {"Fall": 0, "Spring": 1, "Summer": 2, "Winter": 3}

MODEL_PATH = Path("models")
MAX_BATCH_ROWS = 1000
MAX_FILE_SIZE_MB = 10
REQUIRED_BATCH_COLS = [
    "Order_ID",
    "Order_Quantity",
    "Garment_Type",
    "Fabric_Type",
    "Fabric_Width_inches",
    "Pattern_Complexity",
    "Marker_Efficiency_%",
    "Expected_Defect_Rate_%",
    "Operator_Experience_Years",
    "Season",
]
VALID_GARMENTS = set(GARMENT_TYPES)
VALID_FABRICS = set(FABRIC_TYPES)
VALID_COMPLEXITIES = set(PATTERN_COMPLEXITIES)
VALID_SEASONS = set(SEASONS)

MODEL_CLR = {
    "xgboost": "#60A5FA",
    "random_forest": "#34D399",
    "lstm": "#FBBF24",
    "linear_regression": "#F87171",
    "ensemble": "#A78BFA",
}
MODEL_LBL = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lstm": "LSTM",
    "linear_regression": "Linear Regression",
    "ensemble": "Ensemble",
}

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("FabricForecast")

# ============================================================================
# CSS — Professional Design System v4
# ============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp {
    background-color: #0B1120 !important;
    color: #F1F5F9 !important;
}

.stApp > header {
    background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%) !important;
    height: 3px !important; min-height: 3px !important;
}

.stApp footer { visibility: hidden; }
.stApp footer::after {
    visibility: visible; display: block; text-align: center;
    padding: 20px; font-size: .75rem; color: #64748B;
    content: 'Fabric Forecast Pro v3.0.0 \u2014 Azim Mahmud \u00a9 2026';
    border-top: 1px solid #1E293B; margin-top: 40px;
}

h1, h2, h3 { color: #F1F5F9 !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.15rem !important; }

.stMarkdown, .stText, p, span, label, .stCaption { color: #CBD5E1 !important; }
.stCaption { color: #94A3B8 !important; font-size: .85rem !important; }
.stMarkdown strong, .stMarkdown b { color: #F1F5F9 !important; }

a { color: #60A5FA !important; text-decoration: none !important; }
a:hover { color: #93C5FD !important; }

.stMetricLabel {
    color: #94A3B8 !important; font-weight: 600 !important;
    font-size: .8rem !important; text-transform: uppercase !important;
    letter-spacing: .04em !important;
}
.stMetricValue {
    color: #F1F5F9 !important; font-weight: 800 !important;
    font-size: 1.6rem !important;
}
.stMetric [data-testid="stMetricDelta"] { font-weight: 600 !important; }

.stDataFrame td, .stDataFrame th {
    color: #CBD5E1 !important; font-size: .85rem !important;
    background: transparent !important;
}
.stDataFrame th {
    font-weight: 700 !important; text-transform: uppercase !important;
    font-size: .75rem !important; letter-spacing: .04em !important;
    color: #94A3B8 !important; border-bottom: 2px solid #334155 !important;
}
.stDataFrame td { border-bottom: 1px solid #1E293B !important; }

.stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label {
    color: #CBD5E1 !important; font-weight: 600 !important; font-size: .9rem !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    color: #F1F5F9 !important; background: #1E293B !important;
    border: 1.5px solid #334155 !important; border-radius: 8px !important;
}
.stSelectbox svg { fill: #94A3B8 !important; }

.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
    color: #FFFFFF !important; font-weight: 700 !important; font-size: .95rem !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,.45) !important;
}

.stDownloadButton > button {
    background: #1E293B !important; color: #60A5FA !important;
    border: 1.5px solid #334155 !important; font-weight: 700 !important;
    border-radius: 10px !important;
}
.stDownloadButton > button:hover {
    background: #334155 !important; color: #93C5FD !important;
}

.stSuccess { background: rgba(52,211,153,.1) !important; color: #34D399 !important; border-left: 4px solid #34D399 !important; border-radius: 8px !important; }
.stWarning { background: rgba(251,191,36,.1) !important; color: #FBBF24 !important; border-left: 4px solid #FBBF24 !important; border-radius: 8px !important; }
.stError   { background: rgba(248,113,113,.1) !important; color: #F87171 !important; border-left: 4px solid #F87171 !important; border-radius: 8px !important; }
.stInfo    { background: rgba(96,165,250,.1) !important; color: #60A5FA !important; border-left: 4px solid #60A5FA !important; border-radius: 8px !important; }

.stExpander {
    border: 1px solid #334155 !important; border-radius: 10px !important;
    background: #111827 !important;
}
.stExpander header {
    color: #F1F5F9 !important; font-weight: 600 !important;
    background: transparent !important; border: none !important;
}
.stExpander header:hover { background: #1E293B !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 2px !important; border: 1px solid #1E293B !important;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8 !important; font-weight: 600 !important;
    border-radius: 8px !important; padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #1E293B !important; color: #F1F5F9 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.3) !important;
}

div[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.stFileUploader section {
    background: #1E293B !important; border: 2px dashed #334155 !important;
    border-radius: 10px !important; color: #94A3B8 !important;
}

.stProgress > div > div > div { background: linear-gradient(90deg, #3B82F6, #6366F1) !important; border-radius: 10px !important; }

hr { border-color: #1E293B !important; margin: 8px 0 !important; }

.kpi-card {
    background: #111827; border-radius: 12px; padding: 20px 24px;
    border: 1px solid #1E293B;
    box-shadow: 0 1px 3px rgba(0,0,0,.2), 0 1px 2px rgba(0,0,0,.15);
    transition: all .2s ease;
}
.kpi-card:hover {
    border-color: #334155;
    box-shadow: 0 4px 16px rgba(0,0,0,.3);
    transform: translateY(-2px);
}
.kpi-icon {
    width: 44px; height: 44px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin-bottom: 12px;
}
.kpi-label {
    font-size: .7rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.8rem; font-weight: 900; color: #F1F5F9; line-height: 1.1;
}
.kpi-unit {
    font-size: .8rem; font-weight: 700; color: #94A3B8; margin-left: 4px;
}
.kpi-sub {
    font-size: .8rem; font-weight: 600; color: #34D399; margin-top: 8px;
    padding-top: 8px; border-top: 1px solid #1E293B;
}
.kpi-sub.neg { color: #F87171; }

.pred-result {
    background: linear-gradient(135deg, rgba(59,130,246,.12), rgba(99,102,241,.12));
    border: 1.5px solid rgba(96,165,250,.3); border-radius: 14px;
    padding: 28px 32px; text-align: center; margin: 12px 0;
    box-shadow: 0 4px 20px rgba(59,130,246,.1);
}
.pred-result .pred-label {
    font-size: .75rem; font-weight: 700; color: #93C5FD;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px;
}
.pred-result .pred-value {
    font-size: 2.6rem; font-weight: 900; color: #F1F5F9; line-height: 1;
}
.pred-result .pred-unit {
    font-size: .95rem; color: #94A3B8; font-weight: 600;
}
.pred-result .pred-ci {
    font-size: .85rem; color: #93C5FD; margin-top: 10px; font-weight: 600;
}

.rec-box {
    background: rgba(52,211,153,.08);
    border: 1.5px solid rgba(52,211,153,.25); border-radius: 12px;
    padding: 18px 22px; margin: 16px 0;
}
.rec-tag {
    display: inline-block; background: #34D399; color: #0B1120;
    font-size: .65rem; font-weight: 800; padding: 3px 10px;
    border-radius: 6px; letter-spacing: .06em; text-transform: uppercase;
}
.rec-body {
    font-size: .9rem; color: #CBD5E1; margin-top: 8px;
    line-height: 1.6; font-weight: 500;
}

.sidebar-brand {
    text-align: center; padding: 24px 16px 16px 16px;
    border-bottom: 1px solid #1E293B;
}
.sidebar-brand-icon {
    width: 56px; height: 56px; border-radius: 14px;
    background: linear-gradient(135deg, #3B82F6, #6366F1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin: 0 auto 12px auto;
    box-shadow: 0 4px 16px rgba(99,102,241,.3);
}
.sidebar-brand-name {
    font-size: 1.15rem; font-weight: 900; color: #F1F5F9;
    letter-spacing: -0.02em;
}
.sidebar-brand-ver {
    font-size: .7rem; color: #64748B; font-weight: 600; margin-top: 2px;
}

.sidebar-section {
    font-size: .7rem; font-weight: 800; color: #64748B;
    text-transform: uppercase; letter-spacing: .08em;
    padding: 12px 0 6px 0;
}

.sidebar-status {
    border-radius: 8px; padding: 10px 14px;
    text-align: center; font-weight: 700; font-size: .8rem;
    margin: 8px 0;
}
.sidebar-status.prod { background: rgba(52,211,153,.12); color: #34D399; border: 1px solid rgba(52,211,153,.25); }
.sidebar-status.demo { background: rgba(251,191,36,.12); color: #FBBF24; border: 1px solid rgba(251,191,36,.25); }

.sidebar-model-item {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0; font-size: .85rem;
}
.sidebar-model-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}

.page-header { margin-bottom: 24px; }
.page-header-title {
    font-size: 1.6rem; font-weight: 900; color: #F1F5F9;
    letter-spacing: -0.02em; margin-bottom: 4px;
}
.page-header-desc {
    font-size: .9rem; color: #94A3B8; font-weight: 500;
}

.stCodeBlock { border-radius: 10px !important; overflow: hidden !important; }

.stSlider [data-baseweb="slider"] .track { background: #334155 !important; }
.stSlider [data-baseweb="slider"] .inner-track { background: #3B82F6 !important; }
.stSlider [data-testid="stSliderThumbValue"] { color: #F1F5F9 !important; }

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #CBD5E1 !important; background: transparent !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    color: #F1F5F9 !important; background: #1E293B !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #1E293B !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stCaption { color: #64748B !important; }

[data-testid="stSidebarNav"] { display: none !important; }

.stNumberInput input {
    background: #1E293B !important; color: #F1F5F9 !important;
    border: 1.5px solid #334155 !important; border-radius: 8px !important;
}
.stNumberInput input:focus {
    border-color: #3B82F6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,.2) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# HELPERS
# ============================================================================


def _lbl(n):
    return MODEL_LBL.get(n, n.replace("_", " ").title())


def _r2(meta, n):
    v = meta.get("models", {}).get(n, {}).get("r2")
    return f" (R\u00b2={v:.4f})" if isinstance(v, (int, float)) else ""


def _best(meta):
    mm = meta.get("models", {})
    cands = {
        k: v
        for k, v in mm.items()
        if k != "ensemble" and isinstance(v.get("rmse"), (int, float))
    }
    if not cands:
        return "xgboost", mm.get("xgboost", {})
    return min(cands.items(), key=lambda x: x[1]["rmse"])


def _avail(models):
    return [
        k
        for k in ("ensemble", "lstm", "xgboost", "random_forest", "linear_regression")
        if models.get(k) is not None
    ]


def _chart(fig, title="", y_title="", height=420):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#F1F5F9", family="Inter, Arial"),
        ),
        yaxis_title=y_title,
        xaxis_title="",
        plot_bgcolor="#0B1120",
        paper_bgcolor="#0B1120",
        height=height,
        margin=dict(l=60, r=20, t=55, b=45),
        font=dict(family="Inter, Arial", size=12, color="#CBD5E1"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(size=11, color="#94A3B8"),
        title_font=dict(size=12, color="#94A3B8"),
        linecolor="#1E293B",
    )
    fig.update_yaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(size=11, color="#94A3B8"),
        title_font=dict(size=12, color="#94A3B8"),
        linecolor="#1E293B",
    )
    fig.update_traces(
        textfont=dict(color="#CBD5E1", family="Inter, Arial"),
        selector=dict(textposition="outside"),
    )
    return fig


def _hdr(title, desc):
    st.markdown(
        f"""<div class="page-header">
        <div class="page-header-title">{title}</div>
        <div class="page-header-desc">{desc}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()


def _kpi(ico, bg, lbl, val, unit, sub, neg=False):
    cls = "neg" if neg else ""
    u_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    return f"""<div class='kpi-card'>
    <div class='kpi-icon' style='background:{bg}'>{ico}</div>
    <div class='kpi-label'>{lbl}</div>
    <div class='kpi-value'>{val}{u_html}</div>
    <div class='kpi-sub {cls}'>{sub}</div>
    </div>"""


def _validate_str(val, valid_set, field):
    v = str(val).strip()
    if v not in valid_set:
        raise ValueError(f"Invalid {field}: '{v}'. Expected: {sorted(valid_set)}")
    return v


# ============================================================================
# PREDICTION ENGINE
# ============================================================================


def _predict_one(model, X, name):
    X2d = X.reshape(1, -1) if X.ndim != 2 else X
    if name == "lstm" and TF_OK:
        return float(model.predict(X2d.reshape(1, 1, -1), verbose=0).flatten()[0])
    return float(model.predict(X2d)[0])


def predict_with_model(models, meta, X, name):
    model = models.get(name)
    if model is None:
        raise ValueError(f"Model '{_lbl(name)}' not loaded.")
    if name == "ensemble":
        if not isinstance(model, dict):
            raise ValueError("Ensemble data corrupted.")
        weights = model.get("weights", {})
        pred = sum(
            w * _predict_one(models[sub], X, sub)
            for sub, w in weights.items()
            if models.get(sub)
        )
    else:
        pred = _predict_one(model, X, name)
    ci = (
        meta.get("models", {})
        .get(name, {})
        .get("ci_bounds", {})
        .get("ci_fraction", 0.05)
    )
    return {
        "prediction": round(pred, 2),
        "ci_lower": round(pred * (1 - ci), 2),
        "ci_upper": round(pred * (1 + ci), 2),
        "ci_pct": round(ci * 100, 2),
    }


def build_features(
    qty, width_in, eff, defect, exp, garment, fabric, complexity, season, scaler
):
    raw = np.array(
        [
            [
                qty,
                width_in * 2.54,
                eff,
                defect,
                exp,
                GARMENT_ENC[garment],
                FABRIC_ENC[fabric],
                COMPLEXITY_ENC[complexity],
                SEASON_ENC[season],
            ]
        ],
        dtype=np.float32,
    )
    return scaler.transform(raw) if scaler else raw


def compute_bom(qty, garment, width_in, complexity):
    base_m = (
        BASE_CONSUMPTION_M[garment]
        * (160.0 / (width_in * 2.54))
        * COMPLEXITY_MULT[complexity]
    )
    return round(qty * base_m * BOM_BUFFER * METERS_TO_YARDS, 2)


# ============================================================================
# MOCK MODELS
# ============================================================================


class _Mock:
    _B = [
        BASE_CONSUMPTION_M[k] * METERS_TO_YARDS
        for k in ["Dress", "Jacket", "Pants", "Shirt", "T-Shirt"]
    ]
    _C = [1.35, 1.15, 1.00]

    def predict(self, X):
        if X.ndim != 2:
            X = X.reshape(-1, 9)
        g = X[:, 5].astype(int).clip(0, 4)
        c = X[:, 7].astype(int).clip(0, 2)
        base = np.array([self._B[i] for i in g]) * (
            160.0 / np.where(X[:, 1] > 0, X[:, 1], 160)
        )
        base *= np.array([self._C[i] for i in c])
        planned = X[:, 0] * base * BOM_BUFFER
        ef = 1 - (X[:, 2] - 85) / 100 * 0.4
        df = 1 + X[:, 3] / 100
        xf = 1 + np.exp(-X[:, 4] / 15) * 0.04
        return (
            planned
            * ef
            * df
            * xf
            * (1 + np.random.default_rng(42).normal(0, 0.01, len(X)))
        )


class _MockLSTM(_Mock):
    def predict(self, X, verbose=0):
        return super().predict(X).reshape(-1, 1)


# ============================================================================
# MODEL LOADER
# ============================================================================


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    if not JOBLIB_OK or not MODEL_PATH.exists():
        return _demo_pack()
    try:
        p = {}
        for n in (
            "xgboost",
            "random_forest",
            "linear_regression",
            "ensemble",
            "scaler",
            "label_encoders",
        ):
            fp = MODEL_PATH / (
                f"{n}.pkl" if n in ("scaler", "label_encoders") else f"{n}_model.pkl"
            )
            if fp.exists():
                p[n] = joblib.load(fp)
        if TF_OK and (MODEL_PATH / "lstm_model.h5").exists():
            try:
                p["lstm"] = tf.keras.models.load_model(
                    str(MODEL_PATH / "lstm_model.h5"), compile=False
                )
            except Exception as e:
                logger.warning(f"LSTM model load failed: {e}")
        mf = MODEL_PATH / "model_metadata.json"
        meta = json.loads(mf.read_text()) if mf.exists() else _demo_meta()
        if p.get("xgboost") and p.get("scaler"):
            return p, meta, True
    except Exception:
        pass
    return _demo_pack()


def _demo_pack():
    return _demo_models(), _demo_meta(), False


def _demo_models():
    ew = (
        {
            "lstm": 0.51,
            "xgboost": 0.31,
            "random_forest": 0.15,
            "linear_regression": 0.03,
        }
        if TF_OK
        else {"xgboost": 0.50, "random_forest": 0.35, "linear_regression": 0.15}
    )
    return {
        "xgboost": _Mock(),
        "random_forest": _Mock(),
        "linear_regression": _Mock(),
        "lstm": _MockLSTM() if TF_OK else None,
        "ensemble": {"model_names": list(ew), "weights": ew},
        "scaler": None,
    }


def _demo_meta():
    return {
        "version": "3.0.0",
        "unit": "yards",
        "mode": "demo",
        "tensorflow_available": TF_OK,
        "training_date": "2026-04-19",
        "models": {
            "lstm": {
                "rmse": 378.677,
                "mae": 265.233,
                "r2": 0.9955,
                "mape": 5.14,
                "ci_bounds": {"ci_fraction": 0.0768},
            },
            "ensemble": {
                "rmse": 421.509,
                "mae": 287.547,
                "r2": 0.9944,
                "mape": 5.92,
                "ci_bounds": {"ci_fraction": 0.0837},
                "weights": {
                    "lstm": 0.6107,
                    "xgboost": 0.2327,
                    "random_forest": 0.1362,
                    "linear_regression": 0.0204,
                },
            },
            "xgboost": {
                "rmse": 574.154,
                "mae": 404.436,
                "r2": 0.9897,
                "mape": 8.14,
                "ci_bounds": {"ci_fraction": 0.1599},
            },
            "random_forest": {
                "rmse": 821.391,
                "mae": 589.084,
                "r2": 0.9789,
                "mape": 15.92,
                "ci_bounds": {"ci_fraction": 0.249},
            },
            "linear_regression": {
                "rmse": 2062.431,
                "mae": 1567.784,
                "r2": 0.8669,
                "mape": 49.28,
                "ci_bounds": {"ci_fraction": 0.8129},
            },
        },
        "bom_baseline": {"rmse": 484.3, "mae": 323.8, "r2": 0.9927, "mape": 3.86},
        "cv_scores": {
            "xgboost": {
                "rmse_mean": 597.874,
                "rmse_std": 28.308,
                "r2_mean": 0.989,
                "mape_mean": 7.59,
                "mape_std": 0.39,
                "folds_rmse": [624.785, 560.438, 586.944, 636.273, 580.932],
            },
            "random_forest": {
                "rmse_mean": 859.509,
                "rmse_std": 44.354,
                "r2_mean": 0.9772,
                "mape_mean": 16.24,
                "mape_std": 1.41,
                "folds_rmse": [842.696, 855.992, 825.634, 945.454, 827.77],
            },
            "linear_regression": {
                "rmse_mean": 2108.854,
                "rmse_std": 56.323,
                "r2_mean": 0.8627,
                "mape_mean": 50.44,
                "mape_std": 4.77,
                "folds_rmse": [2167.196, 2030.472, 2093.35, 2178.79, 2074.465],
            },
        },
        "feature_importance": {
            "xgboost": {
                "garment_type_encoded": 0.518125,
                "order_quantity": 0.297154,
                "pattern_complexity_encoded": 0.102256,
                "fabric_width_cm": 0.038942,
                "marker_efficiency": 0.010878,
                "defect_rate": 0.008748,
                "operator_experience": 0.009704,
                "fabric_type_encoded": 0.007433,
                "season_encoded": 0.00676,
            },
            "random_forest": {
                "order_quantity": 0.610636,
                "garment_type_encoded": 0.269148,
                "pattern_complexity_encoded": 0.03548,
                "fabric_width_cm": 0.02242,
                "defect_rate": 0.019972,
                "marker_efficiency": 0.018362,
                "operator_experience": 0.012236,
                "fabric_type_encoded": 0.006333,
                "season_encoded": 0.005414,
            },
        },
    }


# ============================================================================
# SIDEBAR
# ============================================================================


def render_sidebar(models, meta, prod):
    with st.sidebar:
        st.markdown(
            """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">&#129533;</div>
            <div class="sidebar-brand-name">Fabric Forecast Pro</div>
            <div class="sidebar-brand-ver">v3.0.0 &middot; yards</div>
        </div>""",
            unsafe_allow_html=True,
        )

        mode_cls = "prod" if prod else "demo"
        mode_txt = "Production" if prod else "Demo Mode"
        st.markdown(
            f'<div class="sidebar-status {mode_cls}">{mode_txt}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section">Navigation</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Nav",
            [
                "Dashboard",
                "Single Predict",
                "Batch Predict",
                "ROI Calculator",
                "Performance",
                "Documentation",
            ],
            label_visibility="collapsed",
        )

        best, _ = _best(meta)
        avail = _avail(models)
        st.markdown(
            '<div class="sidebar-section">Loaded Models</div>',
            unsafe_allow_html=True,
        )
        for m in avail:
            star = " &#9733;" if m == best else ""
            clr = MODEL_CLR.get(m, "#64748B")
            st.markdown(
                f'<div class="sidebar-model-item">'
                f'<div class="sidebar-model-dot" style="background:{clr}"></div>'
                f'<span style="color:#CBD5E1;font-weight:500">{_lbl(m)}</span>'
                f'<span style="color:#64748B;font-size:.75rem">{_r2(meta, m)}</span>'
                f'<span style="color:#FBBF24;font-size:.8rem">{star}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(
            f"TF: {'yes' if TF_OK else 'no'} | Trained: {meta.get('training_date', 'N/A')[:10]}"
        )
    return page


# ============================================================================
# DASHBOARD
# ============================================================================


def page_dashboard(models, meta, prod):
    _hdr("Dashboard", "Model performance overview and operational KPIs")
    mm = meta.get("models", {})
    bom = meta.get("bom_baseline", {})
    best_n, best_m = _best(meta)
    ens = mm.get("ensemble", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _kpi(
                "&#127942;",
                "rgba(96,165,250,.15)",
                "Best Model RMSE",
                f"{best_m.get('rmse', 0):,.1f}",
                "yd",
                _lbl(best_n),
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi(
                "&#127919;",
                "rgba(251,191,36,.15)",
                f"Best R\u00b2 ({_lbl(best_n)})",
                f"{best_m.get('r2', 0):.4f}",
                "",
                "Explained variance",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi(
                "&#128200;",
                "rgba(52,211,153,.15)",
                "Ensemble MAPE",
                f"{ens.get('mape', 0):.2f}",
                "%",
                f"BOM baseline: {bom.get('mape', 0):.2f}%",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        bm_mape = bom.get("mape", 0)
        ens_mape = ens.get("mape", 0)
        imp = round((bm_mape - ens_mape) / bm_mape * 100, 1) if bm_mape else 0
        neg = imp < 0
        st.markdown(
            _kpi(
                "&#128640;" if not neg else "&#128201;",
                "rgba(248,113,113,.15)",
                "MAPE Improvement",
                f"{imp:+.1f}",
                "%",
                "Ensemble vs BOM",
                neg=neg,
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div class='rec-box'><span class='rec-tag'>RECOMMENDATION</span>
    <div class='rec-body'><b>{_lbl(best_n)}</b> has the lowest RMSE ({best_m.get("rmse", 0):,.1f} yd).
    Use it for highest accuracy. For procurement-critical decisions, the <b>Ensemble</b> provides lower variance (RMSE={ens.get("rmse", 0):,.1f} yd).</div></div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    order = ["linear_regression", "random_forest", "lstm", "ensemble", "xgboost"]
    rows = []
    for n in order:
        m = mm.get(n)
        if m:
            rows.append(
                {
                    "Model": _lbl(n),
                    "id": n,
                    "RMSE": m.get("rmse", 0),
                    "MAPE": m.get("mape", 0),
                    "R2": m.get("r2", 0),
                    "Clr": MODEL_CLR.get(n, "#64748B"),
                }
            )
    rows.append(
        {
            "Model": "BOM Baseline",
            "id": "bom",
            "RMSE": bom.get("rmse", 0),
            "MAPE": bom.get("mape", 0),
            "R2": bom.get("r2", 0),
            "Clr": "#64748B",
        }
    )
    dp = pd.DataFrame(rows)

    cl, cr = st.columns(2)
    with cl:
        fig = go.Figure(
            go.Bar(
                x=dp["Model"],
                y=dp["RMSE"],
                marker_color=dp["Clr"].tolist(),
                marker_line_width=0,
                text=[f"{v:,.0f}" for v in dp["RMSE"]],
                textposition="outside",
                textfont=dict(size=12, color="#CBD5E1"),
                width=0.6,
            )
        )
        _chart(fig, "RMSE Comparison", "RMSE (yards)")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig2 = go.Figure(
            go.Bar(
                x=dp["Model"],
                y=dp["MAPE"],
                marker_color=dp["Clr"].tolist(),
                marker_line_width=0,
                text=[f"{v:.1f}" for v in dp["MAPE"]],
                textposition="outside",
                textfont=dict(size=12, color="#CBD5E1"),
                width=0.6,
            )
        )
        _chart(fig2, "MAPE Comparison", "MAPE (%)")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure(
        go.Bar(
            x=dp["Model"],
            y=dp["R2"],
            marker_color=dp["Clr"].tolist(),
            marker_line_width=0,
            text=[f"{v:.4f}" for v in dp["R2"]],
            textposition="outside",
            textfont=dict(size=12, color="#CBD5E1"),
            width=0.6,
        )
    )
    fig3.add_hline(
        y=0.95,
        line_dash="dot",
        line_color="#34D399",
        line_width=1.5,
        annotation_text="0.95 target",
        annotation_font_color="#34D399",
        annotation_font_size=11,
    )
    _chart(fig3, "R\u00b2 Score Comparison", "R\u00b2")
    fig3.update_layout(yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Performance Table")
    display = dp[["Model", "RMSE", "MAPE", "R2"]].rename(
        columns={"RMSE": "RMSE (yd)", "MAPE": "MAPE (%)", "R2": "R\u00b2"}
    )
    st.dataframe(
        display.style.format(
            {"RMSE (yd)": "{:.1f}", "MAPE (%)": "{:.2f}", "R\u00b2": "{:.4f}"}
        ).set_properties(
            **{
                "background-color": "#111827",
                "color": "#CBD5E1",
                "border": "1px solid #1E293B",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Key Findings & Thesis Insights")

    lstm_m  = mm.get("lstm", {})
    xgb_m   = mm.get("xgboost", {})
    rf_m    = mm.get("random_forest", {})
    lr_m    = mm.get("linear_regression", {})
    ens_m   = mm.get("ensemble", {})

    # Build data-driven ranking (all models except ensemble, sorted by RMSE)
    _rank_candidates = [
        (n, mm[n]) for n in ("lstm", "xgboost", "random_forest", "linear_regression")
        if n in mm and isinstance(mm[n].get("rmse"), (int, float))
    ]
    _ranked = sorted(_rank_candidates, key=lambda x: x[1]["rmse"])

    fi1, fi2 = st.columns(2)
    with fi1:
        rank_lines = "\n".join(
            f"{i+1}. **{_lbl(n)}** — {'Best' if i == 0 else f'#{i+1}'} ({v.get('rmse', 0):,.1f} yd RMSE)"
            for i, (n, v) in enumerate(_ranked)
        )
        ens_line = f"\n*(Ensemble: {ens_m.get('rmse', 0):,.1f} yd — combined)*"
        st.info(f"**Model Ranking (by RMSE)**:\n{rank_lines}{ens_line}")
    with fi2:
        bom_rmse    = bom.get("rmse", 0)
        best_rmse   = best_m.get("rmse", 0)
        best_mape   = best_m.get("mape", 0)
        improvement = round((bom_rmse - best_rmse) / bom_rmse * 100, 1) if bom_rmse else 0
        st.warning(f"""
**Critical Note: RMSE vs MAPE**

- BOM baseline has **{bom.get('mape', 0):.2f}% MAPE** (5% safety buffer)
- Best ML ({_lbl(best_n)}) has **{best_mape:.2f}% MAPE**
- BUT: {_lbl(best_n)} has **lower RMSE** ({best_rmse:,.1f} vs {bom_rmse:,.1f} yd)
- RMSE improvement: **{improvement:+.1f}%**

**For thesis**: Use RMSE. MAPE is misleading here because BOM
over-forecasts (5% buffer) while ML predicts actual consumption.
        """)

    st.divider()
    st.subheader("Model Selection Guide")

    sg1, sg2, sg3 = st.columns(3)
    with sg1:
        st.success(f"""
**Best for Accuracy**: {_lbl(best_n)}
- Lowest RMSE ({best_m.get('rmse', 0):,.1f} yd)
- Highest R² ({best_m.get('r2', 0):.4f})
- Use when exact consumption matters
        """)
    with sg2:
        ens_ci = round(ens_m.get("ci_bounds", {}).get("ci_fraction", 0) * 100, 1)
        st.info(f"""
**Best for Robustness**: Ensemble
- Ensemble RMSE: {ens_m.get('rmse', 0):,.1f} yd
- CI width: ±{ens_ci:.1f}% of prediction
- Recommended for procurement decisions
        """)
    with sg3:
        lr_mape = lr_m.get("mape", 0)
        st.error(f"""
**Avoid**: Linear Regression
- Very high MAPE ({lr_mape:.1f}%)
- Misses non-linear patterns
- Only for methodology comparison
        """)

    st.markdown("""
---
*Thesis Note*: The progression Linear Regression → Random Forest → XGBoost → LSTM 
demonstrates that non-linear models progressively capture more fabric consumption 
variance, validating deep learning for this regression problem.
    """)


# ============================================================================
# SINGLE PREDICT
# ============================================================================


def page_single(models, meta, prod):
    _hdr("Single Order Prediction", "Instant forecast with 90% CI and BOM comparison")
    scaler = models.get("scaler")
    cf, cr = st.columns([1, 1], gap="large")
    with cf:
        st.subheader("Order Parameters")
        g_type = st.selectbox("Garment Type", GARMENT_TYPES)
        f_type = st.selectbox("Fabric Type", FABRIC_TYPES)
        qty = st.number_input("Order Quantity", 100, 5000, 1000, 100)
        width_in = st.selectbox(
            "Fabric Width (inches)",
            FABRIC_WIDTHS_INCHES,
            index=2,
            format_func=lambda x: f'{x}" ({round(x * 2.54):.0f} cm)',
        )
        complexity = st.selectbox("Pattern Complexity", PATTERN_COMPLEXITIES, index=1)
        season = st.selectbox("Season", SEASONS)
        eff = st.slider("Marker Efficiency (%)", 70.0, 95.0, 85.0, 0.5)
        defect = st.slider("Expected Defect Rate (%)", 0.0, 10.0, 2.0, 0.1)
        exp = st.slider("Operator Experience (years)", 1, 20, 5)
        avail = _avail(models)
        model_choice = st.selectbox(
            "Model", avail, format_func=lambda x: f"{_lbl(x)}{_r2(meta, x)}"
        )
        btn = st.button("Predict Consumption", type="primary", use_container_width=True)
    with cr:
        st.subheader("Result")
        if btn:
            try:
                X = build_features(
                    qty,
                    width_in,
                    eff,
                    defect,
                    exp,
                    g_type,
                    f_type,
                    complexity,
                    season,
                    scaler,
                )
                r = predict_with_model(models, meta, X, model_choice)
                bom = compute_bom(qty, g_type, width_in, complexity)
                diff = r["prediction"] - bom
                diff_p = diff / bom * 100 if bom else 0
                cost = r["prediction"] * FABRIC_COST_YD[f_type]
                st.markdown(
                    f"""<div class='pred-result'>
                    <div class='pred-label'>Predicted Fabric Consumption</div>
                    <div class='pred-value'>{r["prediction"]:,.2f} <span class='pred-unit'>yards</span></div>
                    <div class='pred-ci'>90% CI: [{r["ci_lower"]:,.2f} &mdash; {r["ci_upper"]:,.2f}] yd (&plusmn;{r["ci_pct"]:.1f}%)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Planned BOM", f"{bom:,.2f} yd")
                m2.metric(
                    "vs BOM",
                    f"{diff:+,.2f} yd",
                    delta=f"{diff_p:+.1f}%",
                    delta_color="inverse",
                )
                m3.metric("Est. Cost", f"${cost:,.0f}")
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=r["prediction"],
                        delta={
                            "reference": bom,
                            "relative": True,
                            "valueformat": ".1%",
                        },
                        gauge={
                            "axis": {
                                "range": [r["ci_lower"] * 0.9, r["ci_upper"] * 1.1],
                                "tickfont": dict(size=11, color="#94A3B8"),
                            },
                            "bar": {"color": "#3B82F6", "thickness": 0.7},
                            "bgcolor": "#0B1120",
                            "steps": [
                                {
                                    "range": [r["ci_lower"] * 0.9, r["ci_lower"]],
                                    "color": "rgba(251,191,36,.1)",
                                },
                                {
                                    "range": [r["ci_lower"], r["ci_upper"]],
                                    "color": "rgba(59,130,246,.1)",
                                },
                                {
                                    "range": [r["ci_upper"], r["ci_upper"] * 1.1],
                                    "color": "rgba(251,191,36,.1)",
                                },
                            ],
                            "threshold": {
                                "line": {"color": "#F87171", "width": 3},
                                "thickness": 0.8,
                                "value": bom,
                            },
                        },
                        title={
                            "text": f"<b>Predicted vs BOM</b><br><span style='font-size:.8rem;color:#94A3B8'>{_lbl(model_choice)}</span>",
                            "font": {"size": 13, "color": "#F1F5F9"},
                        },
                        number={
                            "suffix": " yd",
                            "font": {"size": 24, "color": "#F1F5F9"},
                        },
                    )
                )
                fig.update_layout(
                    height=280,
                    margin=dict(t=65, b=10, l=20, r=20),
                    font=dict(family="Inter, Arial", size=12, color="#CBD5E1"),
                    plot_bgcolor="#0B1120",
                    paper_bgcolor="#0B1120",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Comparison Summary")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Metric": "Predicted (ML)",
                                "Value": f"{r['prediction']:,.2f} yd",
                            },
                            {
                                "Metric": "90% CI Lower",
                                "Value": f"{r['ci_lower']:,.2f} yd",
                            },
                            {
                                "Metric": "90% CI Upper",
                                "Value": f"{r['ci_upper']:,.2f} yd",
                            },
                            {"Metric": "Planned BOM", "Value": f"{bom:,.2f} yd"},
                            {
                                "Metric": "Variance",
                                "Value": f"{diff:+,.2f} yd ({diff_p:+.1f}%)",
                            },
                            {"Metric": "Est. Fabric Cost", "Value": f"${cost:,.0f}"},
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                mi = meta.get("models", {}).get(model_choice, {})
                if mi:
                    with st.expander("Model test-set performance"):
                        _dash = "\u2014"
                        a1, a2, a3, a4 = st.columns(4)
                        a1.metric("RMSE", f"{mi.get('rmse', _dash)} yd")
                        a2.metric("MAE", f"{mi.get('mae', _dash)} yd")
                        a3.metric("R\u00b2", f"{mi.get('r2', _dash)}")
                        a4.metric("MAPE", f"{mi.get('mape', _dash)}%")
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())
        else:
            st.info("Fill in the parameters and click **Predict Consumption**.")


# ============================================================================
# BATCH PREDICT
# ============================================================================


def page_batch(models, meta, prod):
    _hdr("Batch Prediction", "Upload CSV for bulk fabric consumption forecasting")
    scaler = models.get("scaler")
    tmpl = pd.DataFrame(
        [
            (
                "ORD_001",
                1000,
                "T-Shirt",
                "Cotton",
                63,
                "Simple",
                85.0,
                2.0,
                5,
                "Spring",
            ),
            (
                "ORD_002",
                1500,
                "Shirt",
                "Polyester",
                59,
                "Medium",
                88.0,
                3.0,
                8,
                "Summer",
            ),
            ("ORD_003", 2000, "Pants", "Denim", 63, "Complex", 82.0, 4.0, 3, "Fall"),
        ],
        columns=REQUIRED_BATCH_COLS,
    )

    st.subheader("Configuration")
    avail = _avail(models)
    model_choice = st.selectbox(
        "Select Model", avail, format_func=lambda x: f"{_lbl(x)}{_r2(meta, x)}"
    )

    st.markdown(
        """
    <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;padding:20px;margin:12px 0'>
        <div style='font-size:.8rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px'>
            Template Format
        </div>
        <div style='display:flex;flex-wrap:wrap;gap:8px'>
    """,
        unsafe_allow_html=True,
    )
    for col in REQUIRED_BATCH_COLS:
        st.markdown(
            f"<span style='background:#1E293B;color:#CBD5E1;padding:4px 10px;border-radius:6px;font-size:.8rem;font-weight:600'>{col}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1], gap="medium")
    with c1:
        uploaded = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help=f"Max {MAX_FILE_SIZE_MB} MB | {MAX_BATCH_ROWS} rows",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download Template",
            data=tmpl.to_csv(index=False).encode(),
            file_name="batch_template.csv",
            mime="text/csv",
            use_container_width=True,
            key="batch_template_dl",
        )

    if uploaded is None:
        st.info("Upload a CSV file to begin batch prediction.")
        with st.expander("Preview Template"):
            st.dataframe(tmpl, use_container_width=True, hide_index=True)
        return

    if uploaded.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error(f"File too large. Max {MAX_FILE_SIZE_MB} MB.")
        return
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV read error: {e}")
        return
    missing = set(REQUIRED_BATCH_COLS) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(sorted(missing))}")
        return
    if len(df) > MAX_BATCH_ROWS:
        st.warning(f"Truncating to {MAX_BATCH_ROWS} rows.")
        df = df.head(MAX_BATCH_ROWS)
    st.success(f"Loaded **{len(df):,}** orders ready for prediction")

    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.divider()

    results, errors = [], []
    prog = st.progress(0, text="Starting predictions...")
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            w = int(row["Fabric_Width_inches"])
            g = _validate_str(row["Garment_Type"], VALID_GARMENTS, "Garment_Type")
            f = _validate_str(row["Fabric_Type"], VALID_FABRICS, "Fabric_Type")
            cx = _validate_str(
                row["Pattern_Complexity"], VALID_COMPLEXITIES, "Pattern_Complexity"
            )
            s = _validate_str(row["Season"], VALID_SEASONS, "Season")
            q = int(row["Order_Quantity"])
            if q < 100 or q > 5000:
                raise ValueError(f"Quantity {q} out of range (100-5000)")
            X = build_features(
                q,
                w,
                float(row["Marker_Efficiency_%"]),
                float(row["Expected_Defect_Rate_%"]),
                int(row["Operator_Experience_Years"]),
                g,
                f,
                cx,
                s,
                scaler,
            )
            r = predict_with_model(models, meta, X, model_choice)
            bom = compute_bom(q, g, w, cx)
            diff = r["prediction"] - bom
            results.append(
                {
                    "Order_ID": row.get("Order_ID", i),
                    "Garment_Type": g,
                    "Fabric_Type": f,
                    "Order_Quantity": q,
                    "Predicted_yd": r["prediction"],
                    "CI_Lower_yd": r["ci_lower"],
                    "CI_Upper_yd": r["ci_upper"],
                    "Planned_BOM_yd": bom,
                    "Variance_yd": round(diff, 2),
                    "Variance_pct": round(diff / bom * 100, 2) if bom else 0,
                    "Est_Cost_USD": round(r["prediction"] * FABRIC_COST_YD[f], 2),
                }
            )
        except Exception as e:
            errors.append({"Order_ID": row.get("Order_ID", i), "Error": str(e)})
        prog.progress((i + 1) / len(df), text=f"Processing {i + 1}/{len(df)}...")
    prog.empty()

    if errors:
        st.warning(f"**{len(errors)}**/{len(df)} rows had errors.")
        with st.expander("View Errors"):
            st.dataframe(
                pd.DataFrame(errors), use_container_width=True, hide_index=True
            )
    out = pd.DataFrame(results)
    if out.empty:
        st.error("No rows processed successfully.")
        return

    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;color:#F1F5F9;margin:16px 0 8px 0'>"
        f"Results Summary &mdash; {len(out):,} orders processed"
        f"</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            _kpi(
                "&#128230;",
                "rgba(96,165,250,.15)",
                "Orders",
                f"{len(out):,}",
                "",
                f"{len(out) / len(df) * 100:.0f}% success",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            _kpi(
                "&#129533;",
                "rgba(167,139,250,.15)",
                "Total Fabric",
                f"{out['Predicted_yd'].sum():,.0f}",
                "yd",
                f"Avg {out['Predicted_yd'].mean():,.0f} yd/order",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        avg_var = out["Variance_pct"].mean()
        st.markdown(
            _kpi(
                "&#128200;" if avg_var >= 0 else "&#128201;",
                "rgba(52,211,153,.15)" if avg_var >= 0 else "rgba(248,113,113,.15)",
                "Avg Variance",
                f"{avg_var:+.2f}",
                "%",
                "ML vs BOM",
                neg=avg_var < 0,
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            _kpi(
                "&#128176;",
                "rgba(251,191,36,.15)",
                "Total Cost",
                f"${out['Est_Cost_USD'].sum():,.0f}",
                "",
                f"Avg ${out['Est_Cost_USD'].mean():,.0f}/order",
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    tab_charts, tab_table, tab_scatter = st.tabs(
        ["Distribution", "Results Table", "Predicted vs BOM"]
    )

    with tab_charts:
        fig = px.histogram(
            out,
            x="Predicted_yd",
            nbins=30,
            color="Garment_Type",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        _chart(fig, "Consumption Distribution", "Predicted (yards)")
        st.plotly_chart(fig, use_container_width=True)

        fig_cost = px.histogram(
            out,
            x="Est_Cost_USD",
            nbins=25,
            color="Garment_Type",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        _chart(fig_cost, "Cost Distribution", "Est. Cost (USD)")
        st.plotly_chart(fig_cost, use_container_width=True)

    with tab_scatter:
        fig2 = px.scatter(
            out,
            x="Planned_BOM_yd",
            y="Predicted_yd",
            color="Garment_Type",
            hover_data=["Order_ID", "Variance_pct", "Est_Cost_USD"],
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        lo, hi = out["Planned_BOM_yd"].min(), out["Planned_BOM_yd"].max()
        fig2.add_shape(
            type="line",
            x0=lo,
            y0=lo,
            x1=hi,
            y1=hi,
            line=dict(color="#F87171", dash="dot"),
        )
        _chart(fig2, "Predicted vs Planned BOM", "Predicted (yards)")
        fig2.update_layout(xaxis_title="Planned BOM (yards)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_table:
        st.dataframe(
            out.style.format(
                {
                    "Predicted_yd": "{:,.2f}",
                    "CI_Lower_yd": "{:,.2f}",
                    "CI_Upper_yd": "{:,.2f}",
                    "Planned_BOM_yd": "{:,.2f}",
                    "Variance_yd": "{:+,.2f}",
                    "Variance_pct": "{:+.2f}\u0025",
                    "Est_Cost_USD": "${:,.2f}",
                }
            ),
            use_container_width=True,
            height=450,
        )

    st.download_button(
        "Export Results as CSV",
        data=out.to_csv(index=False).encode(),
        file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="batch_results_dl",
    )


# ============================================================================
# ROI CALCULATOR
# ============================================================================


def page_roi(models, meta, prod):
    _hdr("ROI Calculator", "Estimate financial return from switching to ML forecasting")
    bom_b = meta.get("bom_baseline", {})
    best_n, best_m = _best(meta)
    ci, co = st.columns([1, 1.4], gap="large")
    with ci:
        st.subheader("Business Parameters")
        ann = st.number_input("Annual Orders", 100, 100_000, 1000, 100)
        avq = st.number_input("Avg Order Quantity", 100, 10_000, 1500, 100)
        fc = st.number_input("Fabric Cost (USD/yd)", 1.0, 50.0, 7.77, 0.1)
        bm = st.slider("Current BOM MAPE (%)", 1.0, 50.0, bom_b.get("mape", 14.0), 0.5)
        mm_v = st.slider(
            f"ML MAPE (%) \u2014 {_lbl(best_n)}",
            1.0,
            30.0,
            best_m.get("mape", 8.14),
            0.1,
        )
        ic = st.number_input("Implementation Cost (USD)", 0, value=50_000, step=5_000)
        ypc = st.number_input(
            "Avg Yards per Piece",
            min_value=0.5,
            max_value=10.0,
            value=2.34,
            step=0.01,
            help=(
                "Average fabric yards consumed per garment piece. "
                "Derived from garment base consumptions: "
                "T-Shirt≈1.31, Shirt≈1.97, Pants≈2.73, Dress≈3.28, Jacket≈3.83 yd — "
                "weighted average with 5% BOM buffer ≈ 2.34 yd."
            ),
        )
        dr = st.slider("Discount Rate (%)", 1.0, 15.0, 5.0, 0.5)
        ny = st.selectbox("NPV Horizon (years)", [3, 5, 7, 10], index=1)
    saved_yd  = avq * ypc * (bm - mm_v) / 100
    saved_ord = saved_yd * fc
    ann_save = saved_ord * ann
    roi1 = (ann_save - ic) / ic * 100 if ic else 999
    pb = ic / ann_save * 12 if ann_save > 0 else 999
    npv = -ic + sum(ann_save / (1 + dr / 100) ** y for y in range(1, ny + 1))
    with co:
        st.subheader("Analysis")
        r1, r2, r3 = st.columns(3)
        r1.metric("Annual Savings", f"${ann_save:,.0f}")
        r2.metric("Payback", f"{pb:.1f} months")
        r3.metric(f"{ny}-Year NPV", f"${npv:,.0f}")
        st.metric("Year-1 ROI", f"{roi1:.0f}%", help="(Savings - Cost) / Cost x 100")
        st.divider()
        yrs = list(range(ny + 1))
        cum = [-ic + ann_save * y for y in yrs]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=yrs,
                y=cum,
                fill="tozeroy",
                fillcolor="rgba(52,211,153,.08)",
                line=dict(color="#34D399", width=2.5),
            )
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="#F87171",
            line_width=1.5,
            annotation_text="Break-even",
            annotation_font_color="#F87171",
            annotation_font_size=11,
        )
        _chart(fig, "Cumulative Savings", "USD")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Summary")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": "Annual Orders", "Value": f"{ann:,}"},
                    {"Metric": "BOM MAPE", "Value": f"{bm}%"},
                    {"Metric": f"ML MAPE ({_lbl(best_n)})", "Value": f"{mm_v}%"},
                    {"Metric": "Saved / Order", "Value": f"{saved_yd:,.1f} yd"},
                    {"Metric": "$ Saved / Order", "Value": f"${saved_ord:,.2f}"},
                    {"Metric": "Annual Saving", "Value": f"${ann_save:,.0f}"},
                    {"Metric": "Impl. Cost", "Value": f"${ic:,}"},
                    {"Metric": "Year-1 ROI", "Value": f"{roi1:.1f}%"},
                    {"Metric": "Payback", "Value": f"{pb:.1f} months"},
                    {"Metric": f"{ny}-Yr NPV (r={dr}%)", "Value": f"${npv:,.0f}"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================================
# PERFORMANCE
# ============================================================================


def page_perf(models, meta, prod):
    _hdr(
        "Model Performance",
        "Detailed evaluation, cross-validation, and feature analysis",
    )
    mm = meta.get("models", {})
    cv = meta.get("cv_scores", {})
    fi = meta.get("feature_importance", {})
    bom = meta.get("bom_baseline", {})

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Metrics",
            "Model Comparison",
            "Cross-Validation",
            "Feature Importance",
            "CI Coverage",
        ]
    )

    # ── TAB 1: Metrics ──
    with t1:
        st.subheader("Test-Set Performance")
        pr = []
        for n in ("linear_regression", "random_forest", "lstm", "ensemble", "xgboost"):
            m = mm.get(n)
            if m:
                pr.append(
                    {
                        "Model": _lbl(n),
                        "id": n,
                        **{k: m.get(k) for k in ("rmse", "mae", "r2", "mape")},
                    }
                )
        pr.append(
            {
                "Model": "BOM Baseline",
                "id": "bom",
                **{k: bom.get(k) for k in ("rmse", "mae", "r2", "mape")},
            }
        )
        dp = pd.DataFrame(pr).rename(
            columns={
                "rmse": "RMSE (yd)",
                "mae": "MAE (yd)",
                "r2": "R\u00b2",
                "mape": "MAPE (%)",
            }
        )
        styled = (
            dp[["Model", "RMSE (yd)", "MAE (yd)", "R\u00b2", "MAPE (%)"]]
            .style.format(
                {
                    "RMSE (yd)": "{:,.1f}",
                    "MAE (yd)": "{:,.1f}",
                    "R\u00b2": "{:.4f}",
                    "MAPE (%)": "{:.2f}",
                }
            )
            .set_properties(
                **{
                    "background-color": "#111827",
                    "color": "#CBD5E1",
                    "border": "1px solid #1E293B",
                }
            )
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()
        a1, a2 = st.columns(2)
        with a1:
            fig = go.Figure()
            for _, r in dp.iterrows():
                fig.add_trace(
                    go.Bar(
                        name=r["Model"],
                        x=[r["Model"]],
                        y=[r["RMSE (yd)"]],
                        marker_color=MODEL_CLR.get(r["id"], "#64748B"),
                        marker_line_width=0,
                        text=f"{r['RMSE (yd)']:,.0f}",
                        textposition="outside",
                        textfont=dict(size=12, color="#CBD5E1"),
                        width=0.6,
                    )
                )
            _chart(fig, "RMSE by Model", "RMSE (yards)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with a2:
            fig2 = go.Figure()
            for _, r in dp.iterrows():
                fig2.add_trace(
                    go.Bar(
                        name=r["Model"],
                        x=[r["Model"]],
                        y=[r["MAPE (%)"]],
                        marker_color=MODEL_CLR.get(r["id"], "#64748B"),
                        marker_line_width=0,
                        text=f"{r['MAPE (%)']:.2f}",
                        textposition="outside",
                        textfont=dict(size=12, color="#CBD5E1"),
                        width=0.6,
                    )
                )
            _chart(fig2, "MAPE by Model", "MAPE (%)")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: Radar / Comparison ──
    with t2:
        st.subheader("Model Comparison (Normalised Scores)")
        st.caption(
            "Higher is better. Each metric is normalised against the worst model so all axes point in the same direction. "
            "Linear Regression is excluded — its RMSE (~2,062 yd) is an outlier that would visually compress all other models."
        )

        radar_models = []
        for n in ("xgboost", "random_forest", "lstm", "ensemble"):
            m = mm.get(n)
            if m:
                radar_models.append(
                    {
                        "id": n,
                        "name": _lbl(n),
                        "clr": MODEL_CLR.get(n, "#64748B"),
                        "rmse": m.get("rmse", 9999),
                        "mae": m.get("mae", 9999),
                        "r2": m.get("r2", 0),
                        "mape": m.get("mape", 100),
                    }
                )

        if radar_models:
            worst_rmse = max(r["rmse"] for r in radar_models)
            worst_mae = max(r["mae"] for r in radar_models)
            worst_mape = max(r["mape"] for r in radar_models)
            categories = ["RMSE", "MAE", "R\u00b2", "MAPE"]

            fig = go.Figure()
            for r in radar_models:
                rmse_score = 1 - r["rmse"] / worst_rmse if worst_rmse else 0
                mae_score = 1 - r["mae"] / worst_mae if worst_mae else 0
                r2_score = r["r2"]
                mape_score = 1 - r["mape"] / worst_mape if worst_mape else 0
                fig.add_trace(
                    go.Scatterpolar(
                        r=[rmse_score, mae_score, r2_score, mape_score],
                        theta=categories,
                        fill="toself",
                        name=r["name"],
                        line=dict(color=r["clr"], width=2),
                        fillcolor=r["clr"],
                        opacity=0.2,
                    )
                )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=11, color="#94A3B8"),
                        gridcolor="#1E293B",
                        linecolor="#1E293B",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12, color="#CBD5E1", family="Inter, Arial"),
                        gridcolor="#1E293B",
                        linecolor="#1E293B",
                    ),
                    bgcolor="#0B1120",
                ),
                showlegend=True,
                height=450,
                paper_bgcolor="#0B1120",
                plot_bgcolor="#0B1120",
                title=dict(
                    text="<b>Normalised Model Comparison</b>",
                    font=dict(size=15, color="#F1F5F9", family="Inter, Arial"),
                ),
                font=dict(family="Inter, Arial", size=12, color="#CBD5E1"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11, color="#CBD5E1"),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("R\u00b2 Comparison (Full Width)")
        _R2 = "R\u00b2"
        fig3 = go.Figure()
        for _, r in dp.iterrows():
            fig3.add_trace(
                go.Bar(
                    name=r["Model"],
                    x=[r["Model"]],
                    y=[r[_R2]],
                    marker_color=MODEL_CLR.get(r["id"], "#64748B"),
                    marker_line_width=0,
                    text=f"{r[_R2]:.4f}",
                    textposition="outside",
                    textfont=dict(size=12, color="#CBD5E1"),
                    width=0.6,
                )
            )
        fig3.add_hline(
            y=0.95,
            line_dash="dot",
            line_color="#34D399",
            line_width=1.5,
            annotation_text="0.95 target",
            annotation_font_color="#34D399",
            annotation_font_size=11,
        )
        _chart(fig3, "R\u00b2 Score Comparison", "R\u00b2")
        fig3.update_layout(yaxis=dict(range=[0, 1.05]), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3: Cross-Validation ──
    with t3:
        if not cv:
            st.info("No CV data. Run train_models.py first.")
        else:
            st.subheader("5-Fold Cross-Validation Summary")
            cr = []
            for n, v in cv.items():
                cr.append(
                    {
                        "Model": _lbl(n),
                        "id": n,
                        "RMSE Mean": v.get("rmse_mean"),
                        "RMSE Std": v.get("rmse_std"),
                        "R\u00b2 Mean": v.get("r2_mean"),
                        "MAPE Mean": v.get("mape_mean"),
                        "MAPE Std": v.get("mape_std"),
                    }
                )
            dc = pd.DataFrame(cr)
            styled_cv = (
                dc[
                    [
                        "Model",
                        "RMSE Mean",
                        "RMSE Std",
                        "R\u00b2 Mean",
                        "MAPE Mean",
                        "MAPE Std",
                    ]
                ]
                .style.format(
                    {
                        "RMSE Mean": "{:,.1f}",
                        "RMSE Std": "{:.1f}",
                        "R\u00b2 Mean": "{:.4f}",
                        "MAPE Mean": "{:.2f}",
                        "MAPE Std": "{:.2f}",
                    }
                )
                .set_properties(
                    **{
                        "background-color": "#111827",
                        "color": "#CBD5E1",
                        "border": "1px solid #1E293B",
                    }
                )
            )
            st.dataframe(styled_cv, use_container_width=True, hide_index=True)

            st.divider()
            fr = []
            for n, v in cv.items():
                for i, f in enumerate(v.get("folds_rmse", []), 1):
                    fr.append(
                        {"Model": _lbl(n), "id": n, "Fold": f"Fold {i}", "RMSE": f}
                    )
            if fr:
                dff = pd.DataFrame(fr)
                fig = go.Figure()
                for n in cv:
                    s = dff[dff["id"] == n]
                    fig.add_trace(
                        go.Bar(
                            name=_lbl(n),
                            x=s["Fold"],
                            y=s["RMSE"],
                            marker_color=MODEL_CLR.get(n, "#64748B"),
                            marker_line_width=0,
                            text=s["RMSE"].round(1),
                            textposition="outside",
                            textfont=dict(size=11, color="#CBD5E1"),
                        )
                    )
                _chart(fig, "Per-Fold RMSE (5-Fold CV)", "RMSE (yards)")
                fig.update_layout(barmode="group")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("CV RMSE with Error Bars")
                fig2 = go.Figure()
                for _, r in dc.iterrows():
                    fig2.add_trace(
                        go.Bar(
                            name=r["Model"],
                            x=[r["Model"]],
                            y=[r["RMSE Mean"]],
                            marker_color=MODEL_CLR.get(r["id"], "#64748B"),
                            marker_line_width=0,
                            error_y=dict(
                                type="data",
                                array=[r["RMSE Std"]],
                                visible=True,
                                color="#94A3B8",
                            ),
                            text=f"{r['RMSE Mean']:,.1f} &plusmn; {r['RMSE Std']:.1f}",
                            textposition="outside",
                            textfont=dict(size=11, color="#CBD5E1"),
                            width=0.5,
                        )
                    )
                _chart(fig2, "CV RMSE Mean &plusmn; Std", "RMSE (yards)")
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 4: Feature Importance ──
    with t4:
        xf = fi.get("xgboost") or mm.get("xgboost", {}).get("feature_importance", {})
        rf = fi.get("random_forest")
        if not xf and not rf:
            st.info("No feature importance data. Run train_models.py first.")
        else:
            if xf and rf:
                st.subheader("XGBoost vs Random Forest \u2014 Side by Side")
                c1, c2 = st.columns(2)
            else:
                c1 = c2 = st.container()

            if xf:
                with c1 if xf and rf else st.container():
                    if not (xf and rf):
                        st.subheader("XGBoost Feature Importance")
                    dfi = pd.DataFrame(
                        [
                            {"Feature": k.replace("_", " ").title(), "Score": v}
                            for k, v in sorted(
                                xf.items(), key=lambda x: x[1], reverse=True
                            )
                        ]
                    )
                    fig = go.Figure(
                        go.Bar(
                            y=dfi["Feature"][::-1],
                            x=dfi["Score"][::-1],
                            orientation="h",
                            marker_color="#60A5FA",
                            marker_line_width=0,
                            text=dfi["Score"][::-1].round(4),
                            textposition="outside",
                            textfont=dict(size=11, color="#CBD5E1"),
                        )
                    )
                    _chart(fig, "XGBoost", "Score", height=max(320, len(dfi) * 32))
                    st.plotly_chart(fig, use_container_width=True)
            if rf:
                with c2 if xf and rf else st.container():
                    if not (xf and rf):
                        st.subheader("Random Forest Feature Importance")
                    dfr = pd.DataFrame(
                        [
                            {"Feature": k.replace("_", " ").title(), "Score": v}
                            for k, v in sorted(
                                rf.items(), key=lambda x: x[1], reverse=True
                            )
                        ]
                    )
                    fig2 = go.Figure(
                        go.Bar(
                            y=dfr["Feature"][::-1],
                            x=dfr["Score"][::-1],
                            orientation="h",
                            marker_color="#34D399",
                            marker_line_width=0,
                            text=dfr["Score"][::-1].round(4),
                            textposition="outside",
                            textfont=dict(size=11, color="#CBD5E1"),
                        )
                    )
                    _chart(
                        fig2, "Random Forest", "Score", height=max(320, len(dfr) * 32)
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            if xf and rf:
                st.divider()
                st.subheader("Importance Comparison Table")
                all_feats = sorted(set(list(xf.keys()) + list(rf.keys())))
                comp_rows = []
                for feat in all_feats:
                    xv = xf.get(feat, 0)
                    rv = rf.get(feat, 0)
                    comp_rows.append(
                        {
                            "Feature": feat.replace("_", " ").title(),
                            "XGBoost": round(xv, 6),
                            "Random Forest": round(rv, 6),
                            "Diff": round(xv - rv, 6),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(comp_rows)
                    .style.format(
                        {
                            "XGBoost": "{:.6f}",
                            "Random Forest": "{:.6f}",
                            "Diff": "{:+.6f}",
                        }
                    )
                    .set_properties(
                        **{
                            "background-color": "#111827",
                            "color": "#CBD5E1",
                            "border": "1px solid #1E293B",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── TAB 5: CI Coverage ──
    with t5:
        st.subheader("Confidence Interval Width (90th Percentile)")
        ci_rows = []
        for n in ("linear_regression", "random_forest", "lstm", "ensemble", "xgboost"):
            cf = mm.get(n, {}).get("ci_bounds", {}).get("ci_fraction")
            if cf is not None:
                ci_rows.append(
                    {"Model": _lbl(n), "id": n, "CI Width (%)": round(cf * 100, 2)}
                )
        if ci_rows:
            dc = pd.DataFrame(ci_rows)
            fig = go.Figure()
            for _, r in dc.iterrows():
                fig.add_trace(
                    go.Bar(
                        name=r["Model"],
                        x=[r["Model"]],
                        y=[r["CI Width (%)"]],
                        marker_color=MODEL_CLR.get(r["id"], "#6B7280"),
                        marker_line_width=0,
                        text=f"{r['CI Width (%)']:.2f}%",
                        textposition="outside",
                        textfont=dict(size=12, color="#CBD5E1"),
                        width=0.5,
                    )
                )
            fig.add_hline(
                y=5,
                line_dash="dot",
                line_color="#34D399",
                line_width=1.5,
                annotation_text="5% target",
                annotation_font_color="#34D399",
                annotation_font_size=11,
            )
            _chart(fig, "CI Width by Model", "CI Width (%)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                dc[["Model", "CI Width (%)"]]
                .style.format({"CI Width (%)": "{:.2f}"})
                .set_properties(
                    **{
                        "background-color": "#111827",
                        "color": "#CBD5E1",
                        "border": "1px solid #1E293B",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No CI data available.")


# ============================================================================
# DOCUMENTATION
# ============================================================================


def page_docs():
    _hdr("Documentation", "User guide, technical details, and FAQ")
    t1, t2, t3, t4 = st.tabs(["Quick Start", "User Guide", "Technical", "FAQ"])
    with t1:
        st.subheader("Quick Start")
        st.markdown("""
**1.** Install Python 3.10 and dependencies: `pip install -r requirements.txt`

**2.** Generate data: `python data_generation_script.py`

**3.** Train models: `python train_models.py`

**4.** Launch: `streamlit run app.py`
        """)
    with t2:
        st.subheader("Modules")
        st.markdown("""
**Dashboard** \u2014 Auto-detects best model, shows RMSE/MAPE/R\u00b2 charts, recommendation.

**Single Predict** \u2014 Enter parameters, choose model, get prediction with 90% CI and BOM gauge.

**Batch Predict** \u2014 Upload CSV (template provided), validated bulk predictions.

**ROI Calculator** \u2014 Auto-fills BOM and best model MAPE. Savings, ROI, payback, NPV.

**Performance** \u2014 Metrics table, radar comparison, CV with error bars, feature importance, CI.

**Documentation** \u2014 This page.
        """)
        st.subheader("Required Batch CSV Columns")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Column": c, "Type": t, "Example": e}
                    for c, t, e in [
                        ("Order_ID", "String", "ORD_000001"),
                        ("Order_Quantity", "Integer", "1000"),
                        ("Garment_Type", "String", "T-Shirt"),
                        ("Fabric_Type", "String", "Cotton"),
                        ("Fabric_Width_inches", "Integer", "63"),
                        ("Pattern_Complexity", "String", "Simple"),
                        ("Marker_Efficiency_%", "Float", "85.0"),
                        ("Expected_Defect_Rate_%", "Float", "2.0"),
                        ("Operator_Experience_Years", "Integer", "5"),
                        ("Season", "String", "Spring"),
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Garments:** {', '.join(GARMENT_TYPES)}")
            st.markdown(f"**Fabrics:** {', '.join(FABRIC_TYPES)}")
            st.markdown(f"**Widths (in):** {', '.join(map(str, FABRIC_WIDTHS_INCHES))}")
        with c2:
            st.markdown(f"**Complexity:** {', '.join(PATTERN_COMPLEXITIES)}")
            st.markdown(f"**Seasons:** {', '.join(SEASONS)}")
            st.markdown("**Quantity:** 100\u20135,000 | **Efficiency:** 70\u201395%")
            st.markdown("**Defect:** 0\u201310% | **Experience:** 1\u201320 yrs")
    with t3:
        st.subheader("Architecture")

        st.markdown(
            """
        <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;padding:20px;margin-bottom:16px'>
            <div style='font-size:.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>
                Tech Stack
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>
                <div style='background:#1E293B;padding:12px 16px;border-radius:8px'>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>LANGUAGE</div>
                    <div style='font-size:.9rem;color:#F1F5F9;font-weight:700'>Python 3.10</div>
                </div>
                <div style='background:#1E293B;padding:12px 16px;border-radius:8px'>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>WEB FRAMEWORK</div>
                    <div style='font-size:.9rem;color:#F1F5F9;font-weight:700'>Streamlit</div>
                </div>
                <div style='background:#1E293B;padding:12px 16px;border-radius:8px'>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>ML LIBRARIES</div>
                    <div style='font-size:.9rem;color:#F1F5F9;font-weight:700'>scikit-learn, XGBoost, TensorFlow</div>
                </div>
                <div style='background:#1E293B;padding:12px 16px;border-radius:8px'>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>VISUALIZATION</div>
                    <div style='font-size:.9rem;color:#F1F5F9;font-weight:700'>Plotly</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style='font-size:.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;margin:20px 0 10px 0'>
            Model Hyperparameters
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": "Linear Regression",
                        "Estimator": "LinearRegression",
                        "Trees/Epochs": "\u2014",
                        "Max Depth": "\u2014",
                        "Learning Rate": "\u2014",
                    },
                    {
                        "Model": "Random Forest",
                        "Estimator": "RandomForestRegressor",
                        "Trees/Epochs": "300",
                        "Max Depth": "12",
                        "Learning Rate": "\u2014",
                    },
                    {
                        "Model": "XGBoost",
                        "Estimator": "XGBRegressor",
                        "Trees/Epochs": "300",
                        "Max Depth": "8",
                        "Learning Rate": "0.08",
                    },
                    {
                        "Model": "LSTM",
                        "Estimator": "Keras Sequential",
                        "Trees/Epochs": "200",
                        "Max Depth": "\u2014",
                        "Learning Rate": "0.001",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
        <div style='font-size:.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;margin:24px 0 10px 0'>
            LSTM Network Architecture
        </div>
        <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;padding:20px;margin-bottom:16px'>
            <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                <span style='background:#1E293B;color:#60A5FA;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>Input(1, 9)</span>
                <span style='color:#64748B'>&#8594;</span>
                <span style='background:#1E293B;color:#FBBF24;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>LSTM(64)</span>
                <span style='color:#64748B'>&#8594;</span>
                <span style='background:#1E293B;color:#F87171;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>Dropout(0.2)</span>
                <span style='color:#64748B'>&#8594;</span>
                <span style='background:#1E293B;color:#FBBF24;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>LSTM(32)</span>
                <span style='color:#64748B'>&#8594;</span>
                <span style='background:#1E293B;color:#F87171;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>Dropout(0.2)</span>
                <span style='color:#64748B'>&#8594;</span>
                <span style='background:#1E293B;color:#A78BFA;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>Dense(16)</span>
                <span style='color:#64748B'>&#8594;</span>
                <span style='background:#1E293B;color:#34D399;padding:6px 14px;border-radius:8px;font-size:.8rem;font-weight:700'>Dense(1)</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style='font-size:.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;margin:20px 0 10px 0'>
            Feature Pipeline
        </div>
        """,
            unsafe_allow_html=True,
        )
        feat_df = pd.DataFrame(
            [
                {
                    "Feature": "order_quantity",
                    "Type": "int",
                    "Range": "100 \u2013 5,000",
                    "Role": "Volume driver",
                },
                {
                    "Feature": "fabric_width_cm",
                    "Type": "float",
                    "Range": "139.7 \u2013 180.34",
                    "Role": "Efficiency factor",
                },
                {
                    "Feature": "marker_efficiency",
                    "Type": "float",
                    "Range": "70 \u2013 95%",
                    "Role": "Cutting waste",
                },
                {
                    "Feature": "defect_rate",
                    "Type": "float",
                    "Range": "0 \u2013 10%",
                    "Role": "Quality factor",
                },
                {
                    "Feature": "operator_experience",
                    "Type": "int",
                    "Range": "1 \u2013 20 yrs",
                    "Role": "Skill factor",
                },
                {
                    "Feature": "garment_type_encoded",
                    "Type": "int",
                    "Range": "0 \u2013 4",
                    "Role": "Base consumption",
                },
                {
                    "Feature": "fabric_type_encoded",
                    "Type": "int",
                    "Range": "0 \u2013 4",
                    "Role": "Material cost",
                },
                {
                    "Feature": "pattern_complexity_encoded",
                    "Type": "int",
                    "Range": "0 \u2013 2",
                    "Role": "Complexity multiplier",
                },
                {
                    "Feature": "season_encoded",
                    "Type": "int",
                    "Range": "0 \u2013 3",
                    "Role": "Seasonal adjustment",
                },
            ]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
        <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;padding:16px 20px;margin-top:16px'>
            <div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px'>
                <div>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>ENSEMBLE METHOD</div>
                    <div style='font-size:.85rem;color:#CBD5E1;font-weight:600'>Dynamic inverse-RMSE\u00b2 weighting</div>
                </div>
                <div>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>DATA SPLIT</div>
                    <div style='font-size:.85rem;color:#CBD5E1;font-weight:600'>64% train / 16% val / 20% test</div>
                </div>
                <div>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>RANDOM STATE</div>
                    <div style='font-size:.85rem;color:#CBD5E1;font-weight:600'>42</div>
                </div>
                <div>
                    <div style='font-size:.7rem;color:#64748B;font-weight:600'>TARGET UNIT</div>
                    <div style='font-size:.85rem;color:#CBD5E1;font-weight:600'>Yards (1 m = 1.0936 yd)</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with t4:
        st.subheader("FAQ")
        for q, a in [
            (
                "Why Demo Mode?",
                "Models not found. Run `python data_generation_script.py` then `python train_models.py`.",
            ),
            (
                "What is the CI?",
                "90th-percentile empirical interval: 90% of test predictions fell within \u00b1CI%.",
            ),
            (
                "Which model to use?",
                "Dashboard shows recommendation. Ensemble for robustness; best single model for accuracy.",
            ),
            (
                "Can I use metres?",
                "Currently yards-only. Modify TARGET in TrainingConfig to retrain.",
            ),
            (
                "LSTM unavailable?",
                "Install TensorFlow: `pip install tensorflow==2.13.0` then retrain.",
            ),
            (
                "Retrain with new data?",
                "Replace CSV in `generated_data/`, then `python train_models.py`.",
            ),
            (
                "Batch upload column error?",
                "Download the template CSV from the Batch Predict page.",
            ),
        ]:
            with st.expander(q):
                st.markdown(a)


# ============================================================================
# MAIN
# ============================================================================

PAGES = {
    "Dashboard": page_dashboard,
    "Single Predict": page_single,
    "Batch Predict": page_batch,
    "ROI Calculator": page_roi,
    "Performance": page_perf,
    "Documentation": lambda m, mt, p: page_docs(),
}


def main():
    models, meta, prod = load_models()
    page = render_sidebar(models, meta, prod)
    fn = PAGES.get(page)
    if fn:
        fn(models, meta, prod)


if __name__ == "__main__":
    main()