import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PropIQ · House Price Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — Luxury Dark Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #080c14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1280px !important; }

/* ── Animated background mesh ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 900px 600px at 10% 20%, rgba(56,189,248,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 600px 800px at 90% 80%, rgba(168,85,247,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 50% 50%, rgba(20,184,166,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hero Header ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, #f8fafc 0%, #38bdf8 55%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 300;
    letter-spacing: 0.3px;
    max-width: 520px;
    margin: 0 auto 0.5rem;
    line-height: 1.7;
    text-align: center;
}

/* ── Divider ── */
.divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
    margin: 1.5rem 0 2rem;
}

/* ── Stat Cards ── */
.stats-row {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 160px;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg,#38bdf8,#3b82f6));
}
.stat-card:hover { border-color: rgba(56,189,248,0.35); transform: translateY(-2px); }
.stat-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.stat-label { font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 0.4rem; }
.stat-value { font-family: 'DM Mono', monospace; font-size: 1.55rem; font-weight: 500; color: #f1f5f9; }
.stat-unit { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.2rem;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 12px #38bdf8;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
}

/* ── Glass Panel ── */
.glass-panel {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* ── Input area override ── */
.stNumberInput input {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 1rem !important;
}
.stNumberInput input:focus {
    border-color: rgba(56,189,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12) !important;
}
label { color: #94a3b8 !important; font-size: 0.85rem !important; letter-spacing: 0.5px !important; }

/* ── Slider ── */
.stSlider .st-bx { background: rgba(56,189,248,0.2) !important; }
.stSlider .st-by { background: #38bdf8 !important; }

/* ── Predict Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    height: 56px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(56,189,248,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(56,189,248,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Prediction Result ── */
.result-card {
    background: linear-gradient(135deg, rgba(14,165,233,0.12) 0%, rgba(99,102,241,0.12) 100%);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
    animation: fadeUp 0.4s ease;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #a855f7);
}
.result-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38bdf8;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
}
.result-price span {
    background: linear-gradient(135deg, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.5rem;
}
.result-range {
    display: inline-flex;
    gap: 2rem;
    margin-top: 1.5rem;
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
}
.range-item { text-align: center; }
.range-label { font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: #475569; }
.range-value { font-family: 'DM Mono', monospace; font-size: 1rem; color: #94a3b8; margin-top: 2px; }

/* ── Metrics Row ── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.metric-chip {
    flex: 1;
    min-width: 130px;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
}
.metric-chip-label { font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: #475569; margin-bottom: 0.4rem; }
.metric-chip-value { font-family: 'DM Mono', monospace; font-size: 1.3rem; font-weight: 500; }
.metric-good { color: #34d399; }
.metric-mid  { color: #fbbf24; }
.metric-info { color: #38bdf8; }

/* ── Expander override ── */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(148,163,184,0.1) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}
.streamlit-expanderContent {
    background: rgba(8,12,20,0.9) !important;
    border: 1px solid rgba(148,163,184,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

/* ── Success / Info ── */
.stSuccess, .element-container .stSuccess > div {
    background: rgba(52,211,153,0.08) !important;
    border: 1px solid rgba(52,211,153,0.25) !important;
    border-radius: 12px !important;
    color: #34d399 !important;
}

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid rgba(148,163,184,0.08);
    margin-top: 3rem;
}
.footer-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #38bdf8;
    margin-bottom: 0.4rem;
}
.footer-text { font-size: 0.8rem; color: #334155; }

/* ── Keyframe ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glow {
    0%,100% { box-shadow: 0 0 12px rgba(56,189,248,0.4); }
    50%      { box-shadow: 0 0 28px rgba(56,189,248,0.8); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL  (cached for performance)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("area_price_dataset.csv")

@st.cache_resource
def train_model(df):
    X = df[['Area']]
    y = df['Price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "cv_r2": cv_scores.mean(),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return model, scaler, metrics

df = load_data()
model, scaler, metrics = train_model(df)


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">AI-Powered Real Estate Intelligence</div>
    <h1 class="hero-title">PropIQ</h1>
    <div style="display:flex;justify-content:center;width:100%;">
        <p class="hero-sub" style="text-align:center !important;">
            Institutional-grade property valuation powered by machine learning.
            Enter any area size and get an instant, data-driven price estimate.
        </p>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATASET KPI CARDS
# ─────────────────────────────────────────────
avg_price  = df['Price'].mean()
max_area   = df['Area'].max()
total_rows = len(df)
price_range = df['Price'].max() - df['Price'].min()

st.markdown(f"""
<div class="stats-row">
  <div class="stat-card" style="--accent: linear-gradient(90deg,#38bdf8,#0ea5e9)">
    <div class="stat-icon">🏘️</div>
    <div class="stat-label">Properties</div>
    <div class="stat-value">{total_rows:,}</div>
    <div class="stat-unit">training samples</div>
  </div>
  <div class="stat-card" style="--accent: linear-gradient(90deg,#34d399,#0d9488)">
    <div class="stat-icon">📐</div>
    <div class="stat-label">Max Area</div>
    <div class="stat-value">{max_area:,.0f}</div>
    <div class="stat-unit">sq. ft.</div>
  </div>
  <div class="stat-card" style="--accent: linear-gradient(90deg,#fbbf24,#f59e0b)">
    <div class="stat-icon">💎</div>
    <div class="stat-label">Avg. Price</div>
    <div class="stat-value">₹{avg_price:.1f}</div>
    <div class="stat-unit">Lakhs</div>
  </div>
  <div class="stat-card" style="--accent: linear-gradient(90deg,#a855f7,#7c3aed)">
    <div class="stat-icon">📊</div>
    <div class="stat-label">Price Range</div>
    <div class="stat-value">₹{price_range:.1f}</div>
    <div class="stat-unit">Lakhs span</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL METRICS
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <div class="section-dot"></div>
  <div class="section-title">Model Performance</div>
</div>
""", unsafe_allow_html=True)

r2_class   = "metric-good" if metrics["r2"] > 0.85 else "metric-mid"
mae_class  = "metric-info"
rmse_class = "metric-info"

st.markdown(f"""
<div class="metrics-row">
  <div class="metric-chip">
    <div class="metric-chip-label">R² Score</div>
    <div class="metric-chip-value {r2_class}">{metrics['r2']:.4f}</div>
  </div>
  <div class="metric-chip">
    <div class="metric-chip-label">CV R²</div>
    <div class="metric-chip-value metric-good">{metrics['cv_r2']:.4f}</div>
  </div>
  <div class="metric-chip">
    <div class="metric-chip-label">MAE</div>
    <div class="metric-chip-value {mae_class}">{metrics['mae']:.2f}L</div>
  </div>
  <div class="metric-chip">
    <div class="metric-chip-label">RMSE</div>
    <div class="metric-chip-value {rmse_class}">{metrics['rmse']:.2f}L</div>
  </div>
  <div class="metric-chip">
    <div class="metric-chip-label">Status</div>
    <div class="metric-chip-value metric-good">✓ Live</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN LAYOUT  — Two Columns
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.05], gap="large")

# ── LEFT: Input & Prediction ──
with col_left:
    st.markdown("""
    <div class="section-header">
      <div class="section-dot" style="background:#a855f7;box-shadow:0 0 12px #a855f7;"></div>
      <div class="section-title">Price Estimator</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        input_area = st.number_input(
            "Property Area  (sq. ft.)",
            min_value=100,
            max_value=int(df['Area'].max()),
            value=1000,
            step=50,
            help="Enter the total carpet / built-up area of the property."
        )

        predict_btn = st.button("⚡  Estimate Property Value")

    if predict_btn:
        scaled_area   = scaler.transform(pd.DataFrame({'Area': [input_area]}))
        predicted_val = model.predict(scaled_area)[0]

        # Price per sq ft
        ppsf = (predicted_val * 100_000) / input_area

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-price">₹ <span>{predicted_val:.2f}</span> Lakhs</div>
            <div class="result-sub">Based on {total_rows} comparable transactions</div>
            <div class="result-range">
                <div class="range-item">
                    <div class="range-label">Price / sq.ft</div>
                    <div class="range-value">₹ {ppsf:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mini bar showing where input sits in distribution ──
        percentile = (df['Area'] < input_area).mean() * 100
        st.markdown(f"""
        <div style="margin-top:1.2rem;background:rgba(15,23,42,0.7);border:1px solid rgba(148,163,184,0.1);
                    border-radius:14px;padding:1rem 1.4rem;">
            <div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#475569;margin-bottom:0.6rem;">
                Area Percentile vs Dataset
            </div>
            <div style="background:rgba(30,41,59,0.8);border-radius:8px;height:8px;overflow:hidden;margin-bottom:0.4rem;">
                <div style="height:100%;width:{percentile:.0f}%;
                            background:linear-gradient(90deg,#0ea5e9,#a855f7);
                            border-radius:8px;transition:width 0.6s ease;"></div>
            </div>
            <div style="font-size:0.82rem;color:#64748b;">
                This property is larger than <b style="color:#94a3b8">{percentile:.0f}%</b> of properties in the dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Regression Chart ──
with col_right:
    st.markdown("""
    <div class="section-header">
      <div class="section-dot" style="background:#34d399;box-shadow:0 0 12px #34d399;"></div>
      <div class="section-title">Regression Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0b1322')
    ax.set_facecolor('#0b1322')

    # Scatter
    ax.scatter(
        df["Area"], df["Price"],
        color="#38bdf8", alpha=0.55, s=28, zorder=3,
        label="Transactions", edgecolors='none'
    )

    # Regression line
    area_range = np.linspace(df["Area"].min(), df["Area"].max(), 200)
    area_sc    = scaler.transform(pd.DataFrame({'Area': area_range}))
    price_line = model.predict(area_sc)
    ax.plot(area_range, price_line, color="#a855f7", linewidth=2.5, zorder=4, label="Regression Line")

    # If prediction was made
    if predict_btn:
        ax.axvline(input_area, color="#fbbf24", linewidth=1.4, linestyle="--", alpha=0.7)
        ax.scatter([input_area], [predicted_val],
                   color="#fbbf24", s=120, zorder=6, edgecolors="#0b1322", linewidth=2,
                   label=f"Your Input")

    # Styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.set_xlabel("Area  (sq. ft.)", color="#64748b", fontsize=10, labelpad=8)
    ax.set_ylabel("Price  (Lakhs ₹)", color="#64748b", fontsize=10, labelpad=8)
    ax.set_title("Area vs Price  —  Linear Regression", color="#94a3b8", fontsize=11, pad=14)
    ax.grid(True, color='#1e293b', linewidth=0.8, linestyle='--')
    legend = ax.legend(facecolor='#0f172a', edgecolor='#1e293b', labelcolor='#94a3b8', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────
#  DATA EXPLORER  (collapsible)
# ─────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

with st.expander("📂  Raw Dataset Explorer"):
    tab1, tab2 = st.tabs(["  Table View  ", "  Statistics  "])
    with tab1:
        st.dataframe(df.style.background_gradient(cmap="Blues", subset=["Price"]), use_container_width=True)
    with tab2:
        st.dataframe(df.describe().round(2), use_container_width=True)

with st.expander("🔬  Model — Actual vs Predicted (Test Set)"):
    comparison_df = pd.DataFrame({
        "Actual Price (₹L)":    metrics["y_test"].values,
        "Predicted Price (₹L)": np.round(metrics["y_pred"], 2),
        "Error (₹L)":           np.round(metrics["y_test"].values - metrics["y_pred"], 2),
        "Error %":              np.round(
            (metrics["y_test"].values - metrics["y_pred"]) / metrics["y_test"].values * 100, 2
        )
    })
    st.dataframe(
        comparison_df.style
            .background_gradient(cmap="RdYlGn", subset=["Error %"], vmin=-20, vmax=20),
        use_container_width=True
    )


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-wrap">
    <div class="footer-logo">PropIQ</div>
    <div class="footer-text">
        Built with Streamlit · scikit-learn · matplotlib &nbsp;|&nbsp;
        Linear Regression · StandardScaler · Cross-Validation
    </div>
</div>
""", unsafe_allow_html=True)