import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import io

# reportlab for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    --accent:      #6C63FF;
    --accent-2:    #FF6584;
    --success:     #22c55e;
    --danger:      #ef4444;
    --warning:     #f59e0b;
    --radius:      14px;
    --card-shadow: 0 4px 24px rgba(0,0,0,.10);
    --transition:  .3s cubic-bezier(.4,0,.2,1);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

@media (prefers-color-scheme: light) { .stApp { background: #f4f6ff; } }
@media (prefers-color-scheme: dark)  { .stApp { background: #0f1117; } }

/* ── Animations ── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes resultPop {
    0%   { opacity: 0; transform: scale(.92) translateY(14px); }
    60%  { transform: scale(1.02) translateY(-3px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes barGrow { from { width: 0%; } }

/* ── Hero ── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700;
    background: linear-gradient(270deg, #6C63FF, #FF6584, #38bdf8, #6C63FF);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease infinite, fadeSlideDown .8s ease both;
    margin: 0 0 4px; line-height: 1.15;
}
.hero-sub {
    color: #94a3b8; font-size: 1.05rem; font-weight: 400;
    animation: fadeSlideDown 1s ease .2s both; margin-bottom: 0;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, var(--accent) 0%, #9b8cff 100%);
    border-radius: var(--radius); padding: 22px 20px 18px;
    color: #fff; text-align: center; box-shadow: var(--card-shadow);
    animation: fadeSlideUp .7s ease both;
    transition: transform var(--transition), box-shadow var(--transition);
    cursor: default;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 10px 32px rgba(108,99,255,.35); }
.metric-card .val  { font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; line-height: 1; letter-spacing: -.5px; }
.metric-card .lbl  { font-size: .82rem; font-weight: 500; opacity: .85; margin-top: 5px; text-transform: uppercase; letter-spacing: .06em; }
.metric-card.green { background: linear-gradient(135deg, #16a34a, #22c55e); }
.metric-card.red   { background: linear-gradient(135deg, #b91c1c, #ef4444); }
.metric-card.amber { background: linear-gradient(135deg, #b45309, #f59e0b); }
.metric-card.blue  { background: linear-gradient(135deg, #1d4ed8, #38bdf8); }

/* ── Section headers ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 600; letter-spacing: -.2px;
    padding: 6px 0 10px; border-bottom: 2px solid #6C63FF33;
    margin-bottom: 18px; animation: fadeSlideUp .6s ease both;
}

/* ── Result banners ── */
.result-churn {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 50%, #b91c1c 100%);
    border: 1px solid #ef4444; border-radius: var(--radius);
    padding: 24px 28px; color: #fef2f2;
    font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600;
    animation: resultPop .5s ease both; box-shadow: 0 6px 32px rgba(239,68,68,.25);
}
.result-safe {
    background: linear-gradient(135deg, #14532d 0%, #15803d 50%, #16a34a 100%);
    border: 1px solid #22c55e; border-radius: var(--radius);
    padding: 24px 28px; color: #f0fdf4;
    font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600;
    animation: resultPop .5s ease both; box-shadow: 0 6px 32px rgba(34,197,94,.25);
}

/* ── Probability bar ── */
.prob-bar-wrap { background: #1e293b; border-radius: 99px; height: 12px; margin-top: 12px; overflow: hidden; }
.prob-bar { height: 100%; border-radius: 99px; animation: barGrow .9s cubic-bezier(.4,0,.2,1) both; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label { color: #c7d2fe !important; font-weight: 500; }

/* ── FIX: pointer cursor on ALL dropdown/select elements ── */
div[data-baseweb="select"],
div[data-baseweb="select"] *,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] svg,
.stSelectbox,
.stSelectbox > div,
.stSelectbox > div > div,
.stSelectbox > div > div > div,
.stSelectbox svg,
[role="listbox"],
[role="option"],
[role="combobox"],
[data-testid="stSelectbox"] * {
    cursor: pointer !important;
}
/* +/- buttons on number inputs */
.stNumberInput button,
.stNumberInput button svg,
button[aria-label="Decrement"],
button[aria-label="Increment"] {
    cursor: pointer !important;
}

/* ── Input widgets — visible in BOTH light & dark mode ── */

/* Selectbox container */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div > div {
    background-color: #2d2b6b !important;
    border: 1.5px solid #5b52d6 !important;
    border-radius: 9px !important;
    color: #e0e7ff !important;
    transition: box-shadow var(--transition), border-color var(--transition) !important;
}
/* Selectbox selected value + placeholder text */
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="placeholder"],
section[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #e0e7ff !important;
    -webkit-text-fill-color: #e0e7ff !important;
}
/* Dropdown arrow SVG */
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #a5b4fc !important;
    color: #a5b4fc !important;
}
/* Dropdown list popup */
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul {
    background-color: #1e1b4b !important;
    border: 1px solid #5b52d6 !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] {
    color: #e0e7ff !important;
    background-color: #1e1b4b !important;
}
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
}

/* Number input box */
section[data-testid="stSidebar"] .stNumberInput > div > div {
    background-color: #2d2b6b !important;
    border: 1.5px solid #5b52d6 !important;
    border-radius: 9px !important;
}
section[data-testid="stSidebar"] .stNumberInput input {
    background-color: #2d2b6b !important;
    color: #e0e7ff !important;
    -webkit-text-fill-color: #e0e7ff !important;
    font-weight: 500 !important;
}
/* Number input +/- buttons */
section[data-testid="stSidebar"] .stNumberInput button {
    background-color: #3d3a8c !important;
    color: #a5b4fc !important;
    border: none !important;
}
section[data-testid="stSidebar"] .stNumberInput button:hover {
    background-color: #4f46e5 !important;
    color: #fff !important;
}

/* Focus glow */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within {
    box-shadow: 0 0 0 3px rgba(108,99,255,.45) !important;
    border-color: #818cf8 !important;
}

/* ── Predict button ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; letter-spacing: .03em !important;
    background: linear-gradient(135deg, #6C63FF, #9b8cff) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 14px 32px !important;
    width: 100% !important; cursor: pointer !important;
    transition: transform var(--transition), box-shadow var(--transition) !important;
    box-shadow: 0 4px 18px rgba(108,99,255,.35) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(108,99,255,.5) !important; }

/* ── Download buttons ── */
.stDownloadButton > button {
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 12px 28px !important;
    width: 100% !important; cursor: pointer !important;
    transition: transform var(--transition), box-shadow var(--transition) !important;
    margin-top: 10px;
    background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
    box-shadow: 0 4px 18px rgba(20,184,166,.3) !important;
}
.stDownloadButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(0,0,0,.25) !important; }

/* ── Plotly ── */
.stPlotlyChart { animation: fadeSlideUp .7s ease both; }

/* ── Divider ── */
.fancy-divider {
    height: 3px; border-radius: 99px;
    background: linear-gradient(90deg, #6C63FF, #FF6584, #38bdf8);
    margin: 28px 0; animation: shimmer 3s linear infinite; background-size: 200% auto;
}

/* ── About card ── */
.about-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #1e1b4b 100%);
    border: 1px solid #4f46e580; border-radius: 20px;
    padding: 36px 40px; max-width: 700px;
    margin: 20px auto; text-align: center;
    animation: fadeSlideUp .8s ease both;
    box-shadow: 0 8px 40px rgba(108,99,255,.25);
}
.about-avatar {
    width: 110px; height: 110px; border-radius: 50%;
    border: 4px solid #6C63FF;
    background: linear-gradient(135deg, #6C63FF, #FF6584);
    display: flex; align-items: center; justify-content: center;
    font-size: 3.2rem; margin: 0 auto 18px;
    box-shadow: 0 4px 24px rgba(108,99,255,.4);
}
.about-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #e0e7ff; margin-bottom: 4px; }
.about-role { color: #a5b4fc; font-size: 1rem; font-weight: 500; margin-bottom: 22px; }
.about-divider { height: 1px; background: #4f46e560; margin: 18px 0; }
.about-contact a { color: #a5b4fc !important; text-decoration: none; font-size: .95rem; font-weight: 500; transition: color .2s; }
.about-contact a:hover { color: #c7d2fe !important; text-decoration: underline; }
.about-badge {
    display: inline-block; background: #312e81; border: 1px solid #4f46e5;
    border-radius: 99px; padding: 7px 18px;
    font-size: .82rem; font-weight: 600; color: #c7d2fe; margin: 5px 4px;
    transition: background var(--transition);
}
.about-badge:hover { background: #4f46e5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA + MODEL  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=450, learning_rate=0.01, max_depth=3,
        subsample=0.7, colsample_bytree=0.7, gamma=0.3,
        reg_lambda=2, reg_alpha=0.5, random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    return df, X, y, model, encoders, x_test, y_test, y_pred, y_prob, acc


df, X, y, model, encoders, x_test, y_test, y_pred, y_prob, acc = load_and_train()
report    = classification_report(y_test, y_pred, output_dict=True)
precision = report["1"]["precision"]
recall    = report["1"]["recall"]
f1        = report["1"]["f1-score"]
churn_rate = y.mean()


# ─────────────────────────────────────────────
#  PDF GENERATOR  (reportlab)
# ─────────────────────────────────────────────
def generate_pdf_report(result_data, prediction, churn_prob, safe_prob,
                         risk_label, acc, precision, recall, f1):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    PURPLE  = colors.HexColor("#6C63FF")
    DARK    = colors.HexColor("#1e1b4b")
    LIGHT   = colors.HexColor("#e0e7ff")
    RED_C   = colors.HexColor("#ef4444")
    GREEN_C = colors.HexColor("#22c55e")
    GREY    = colors.HexColor("#94a3b8")
    LAVENDER= colors.HexColor("#ede9fe")
    STRIPE  = colors.HexColor("#f5f3ff")
    BORDER  = colors.HexColor("#c4b5fd")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleS", parent=styles["Title"],
        textColor=PURPLE, fontSize=22, spaceAfter=4, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("SubS", parent=styles["Normal"],
        textColor=GREY, fontSize=10, spaceAfter=16, fontName="Helvetica")
    h2_style = ParagraphStyle("H2S", parent=styles["Heading2"],
        textColor=DARK, fontSize=13, spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    footer_style = ParagraphStyle("FootS", parent=styles["Normal"],
        fontSize=8, textColor=GREY, fontName="Helvetica", alignment=1)

    story = []

    # ── Title
    story.append(Paragraph("Customer Churn Prediction Report", title_style))
    story.append(Paragraph("Generated by Churn Predictor App  |  Author: Sanjay Choudhari", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=14))

    # ── Prediction result
    story.append(Paragraph("Prediction Result", h2_style))
    outcome_color = RED_C if prediction[0] == 1 else GREEN_C
    outcome_text  = "WILL CHURN" if prediction[0] == 1 else "WILL NOT CHURN"
    result_rows = [
        ["Outcome", outcome_text],
        ["Churn Probability",     f"{churn_prob:.4f}  ({churn_prob:.1%})"],
        ["Retention Probability", f"{safe_prob:.4f}  ({safe_prob:.1%})"],
        ["Risk Level",            risk_label],
    ]
    rt = Table(result_rows, colWidths=[6*cm, 11*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, -1), LAVENDER),
        ("BACKGROUND",   (1, 0), (1,  0), outcome_color),
        ("TEXTCOLOR",    (1, 0), (1,  0), colors.white),
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",     (0, 0), (0,  -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [STRIPE, colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(rt)
    story.append(Spacer(1, 16))

    # ── Model performance
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=10))
    story.append(Paragraph("Model Performance", h2_style))
    perf_rows = [
        ["Metric",    "Value"],
        ["Accuracy",  f"{acc:.4f}  ({acc:.1%})"],
        ["Precision", f"{precision:.4f}"],
        ["Recall",    f"{recall:.4f}"],
        ["F1 Score",  f"{f1:.4f}"],
    ]
    pt = Table(perf_rows, colWidths=[6*cm, 11*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1,-1), [STRIPE, colors.white]),
        ("GRID",         (0, 0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0, 0), (-1,-1), 8),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 8),
        ("LEFTPADDING",  (0, 0), (-1,-1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 16))

    # ── Input features
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=10))
    story.append(Paragraph("Input Features", h2_style))
    feat_df   = result_data.drop(["Prediction", "Churn_Probability", "Risk_Level"], axis=1)
    feat_rows = [["Feature", "Value"]] + [[str(c), str(feat_df[c].iloc[0])] for c in feat_df.columns]
    ft = Table(feat_rows, colWidths=[8*cm, 9*cm])
    ft.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), LIGHT),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1,-1), [STRIPE, colors.white]),
        ("GRID",         (0, 0), (-1,-1), 0.4, BORDER),
        ("TOPPADDING",   (0, 0), (-1,-1), 6),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 6),
        ("LEFTPADDING",  (0, 0), (-1,-1), 10),
    ]))
    story.append(ft)
    story.append(Spacer(1, 20))

    # ── Footer
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=8))
    story.append(Paragraph(
        "Sanjay Choudhari  |  sanjaychoudhari288@gmail.com  |  github.com/Sanjaymo  |  +91 9963785768",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ─────────────────────────────────────────────
#  NAVIGATION  (sidebar radio)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Navigation")
    page = st.radio("", ["🏠 Dashboard", "🔮 Predict", "👤 About"], label_visibility="collapsed")
    st.markdown("---")


# ═════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<h1 class="hero-title">📡 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">XGBoost-powered intelligence · Real-time risk scoring · Interactive analytics</p>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card blue"><div class="val">{acc:.1%}</div><div class="lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card amber"><div class="val">{churn_rate:.1%}</div><div class="lbl">Churn Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card green"><div class="val">{precision:.2f}</div><div class="lbl">Precision</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card red"><div class="val">{recall:.2f}</div><div class="lbl">Recall</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Model Analytics Dashboard</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Feature Importance", "🧩 Confusion Matrix"])

    with tab1:
        ca, cb = st.columns(2)
        with ca:
            churn_counts = y.value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            churn_counts["Label"] = churn_counts["Churn"].map({0: "No Churn", 1: "Churn"})
            fig_donut = px.pie(
                churn_counts, values="Count", names="Label", hole=0.55, color="Label",
                color_discrete_map={"No Churn": "#6C63FF", "Churn": "#FF6584"},
                title="Churn Distribution"
            )
            fig_donut.update_traces(textposition="outside", textinfo="percent+label")
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False, title_font=dict(family="Space Grotesk", size=16))
            st.plotly_chart(fig_donut, use_container_width=True)
        with cb:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=y_prob[y_test == 0], name="No Churn", marker_color="#6C63FF", opacity=0.75, nbinsx=30))
            fig_hist.add_trace(go.Histogram(x=y_prob[y_test == 1], name="Churn",    marker_color="#FF6584", opacity=0.75, nbinsx=30))
            fig_hist.update_layout(barmode="overlay", title="Predicted Probability Distribution",
                xaxis_title="Churn Probability", yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", legend=dict(font=dict(color="#94a3b8")),
                title_font=dict(family="Space Grotesk", size=16))
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)
        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#312e81", "#6C63FF", "#FF6584"],
            title="Top 15 Feature Importances")
        fig_feat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False,
            title_font=dict(family="Space Grotesk", size=16), coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    with tab3:
        cm_mat = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm_mat, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["No Churn", "Churn"], y=["No Churn", "Churn"],
            color_continuous_scale=["#1e1b4b", "#6C63FF", "#FF6584"], title="Confusion Matrix")
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", title_font=dict(family="Space Grotesk", size=16))
        st.plotly_chart(fig_cm, use_container_width=True)


# ═════════════════════════════════════════════
#  PAGE 2 — PREDICT
# ═════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<h1 class="hero-title">🔮 Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Fill in customer details in the sidebar, then click Predict</p>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar customer form
    with st.sidebar:
        st.markdown("## 🧑‍💼 Customer Profile")
        st.markdown("Fill the fields and click **Predict** on the main panel.")
        st.markdown("---")

        user_data = {}
        for col in X.columns:
            if col in encoders:
                options = list(encoders[col].classes_)
                value = st.selectbox(col, options, key=f"sb_{col}")
                user_data[col] = encoders[col].transform([value])[0]
            else:
                # Detect integer-only columns and use step=1 / format="%d" to avoid decimals
                col_vals = df[col].dropna()
                is_integer_col = col_vals.apply(lambda v: float(v) == int(float(v))).all()
                if is_integer_col:
                    value = st.number_input(col, value=0, step=1, format="%d", key=f"ni_{col}")
                else:
                    value = st.number_input(col, value=0.0, step=0.01, key=f"ni_{col}")
                user_data[col] = value

    input_df = pd.DataFrame([user_data])

    st.markdown('<div class="section-header">⚡ Run Prediction</div>', unsafe_allow_html=True)

    if st.button("⚡  Predict Churn Risk", use_container_width=True):
        prediction = model.predict(input_df)
        prob       = model.predict_proba(input_df)
        churn_prob = prob[0][1]
        safe_prob  = prob[0][0]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-churn">
                🚨 &nbsp; Customer <strong>WILL CHURN</strong>
                <div style="font-size:.9rem;font-weight:400;margin-top:6px;opacity:.85">
                    Churn probability: <strong>{churn_prob:.1%}</strong>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{churn_prob*100:.1f}%;background:linear-gradient(90deg,#ef4444,#fca5a5);"></div>
                </div>
            </div>""", unsafe_allow_html=True)
            risk_label = "High Risk"
        else:
            st.markdown(f"""
            <div class="result-safe">
                ✅ &nbsp; Customer will <strong>NOT CHURN</strong>
                <div style="font-size:.9rem;font-weight:400;margin-top:6px;opacity:.85">
                    Retention probability: <strong>{safe_prob:.1%}</strong>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{safe_prob*100:.1f}%;background:linear-gradient(90deg,#22c55e,#86efac);"></div>
                </div>
            </div>""", unsafe_allow_html=True)
            risk_label = "Low Risk"

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 32, "family": "Space Grotesk"}},
            title={"text": "Churn Probability", "font": {"size": 16, "family": "Space Grotesk", "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#6C63FF"}, "bgcolor": "#1e293b",
                "steps": [
                    {"range": [0,  40], "color": "#14532d"},
                    {"range": [40, 70], "color": "#713f12"},
                    {"range": [70,100], "color": "#7f1d1d"},
                ],
                "threshold": {"line": {"color": "#FF6584", "width": 3}, "thickness": 0.75, "value": churn_prob * 100},
            },
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
            height=280, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Download section
        st.markdown('<div class="section-header">⬇️ Download Prediction Report</div>', unsafe_allow_html=True)

        result_data = input_df.copy()
        result_data["Prediction"]        = "Churn" if prediction[0] == 1 else "No Churn"
        result_data["Churn_Probability"] = round(float(churn_prob), 4)
        result_data["Risk_Level"]        = risk_label

        dl1, dl2 = st.columns(2)

        with dl1:
            csv_buffer = io.StringIO()
            result_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥  Download as CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_prediction_result.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl2:
            pdf_bytes = generate_pdf_report(
                result_data, prediction, churn_prob, safe_prob,
                risk_label, acc, precision, recall, f1
            )
            st.download_button(
                label="📄  Download PDF Report",
                data=pdf_bytes,
                file_name="churn_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ═════════════════════════════════════════════
#  PAGE 3 — ABOUT
# ═════════════════════════════════════════════
elif page == "👤 About":
    st.markdown('<h1 class="hero-title">👤 About the Author</h1>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="about-avatar">👨‍💻</div>
        <div class="about-name">Sanjay Choudhari</div>
        <div class="about-role">Machine Learning Engineer &amp; Data Scientist</div>
        <div class="about-divider"></div>
        <div style="text-align:left; padding: 0 10px;">
            <p style="color:#c7d2fe; font-size:.95rem; line-height:1.75; margin-bottom:16px;">
                Passionate about building intelligent systems that turn raw data into real-world decisions.
                This Customer Churn Prediction app uses an XGBoost classifier trained on the Telco dataset
                to help businesses identify at-risk customers before they leave — with interactive analytics
                and downloadable reports.
            </p>
        </div>
        <div class="about-divider"></div>
        <div style="margin-bottom:18px;">
            <span class="about-badge">🤖 Machine Learning</span>
            <span class="about-badge">📊 Data Science</span>
            <span class="about-badge">🐍 Python</span>
            <span class="about-badge">⚡ XGBoost</span>
            <span class="about-badge">🚀 Streamlit</span>
            <span class="about-badge">📈 Plotly</span>
        </div>
        <div class="about-divider"></div>
        <div class="about-contact" style="display:flex; flex-direction:column; gap:12px; align-items:center; margin-top:16px;">
            <a href="mailto:sanjaychoudhari288@gmail.com">📧 &nbsp; sanjaychoudhari288@gmail.com</a>
            <a href="https://github.com/Sanjaymo" target="_blank">🐙 &nbsp; github.com/Sanjaymo</a>
            <a href="tel:+919963785768">📞 &nbsp; +91 9963785768</a>
        </div>
        <div class="about-divider"></div>
        <p style="color:#6366f1; font-size:.82rem; margin-top:12px;">
            Built with ❤️ using Python · Streamlit · XGBoost · Plotly · ReportLab
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project stats row
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📦 Project Stats</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="metric-card blue"><div class="val">{len(df):,}</div><div class="lbl">Total Records</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="metric-card amber"><div class="val">{X.shape[1]}</div><div class="lbl">Features Used</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="metric-card green"><div class="val">{acc:.1%}</div><div class="lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="metric-card red"><div class="val">{f1:.2f}</div><div class="lbl">F1 Score</div></div>', unsafe_allow_html=True)
