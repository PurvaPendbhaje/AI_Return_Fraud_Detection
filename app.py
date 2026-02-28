import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

:root {
    --bg:        #F7F8FA;
    --surface:   #FFFFFF;
    --border:    #E4E7EC;
    --border2:   #D0D5DD;
    --text1:     #101828;
    --text2:     #475467;
    --text3:     #98A2B3;
    --accent:    #1570EF;
    --success:   #12B76A;
    --success-l: #ECFDF3;
    --warn:      #F79009;
    --warn-l:    #FFFAEB;
    --danger:    #F04438;
    --danger-l:  #FEF3F2;
    --shadow-s:  0 1px 3px rgba(16,24,40,.08), 0 1px 2px rgba(16,24,40,.04);
    --shadow-m:  0 4px 8px -2px rgba(16,24,40,.10), 0 2px 4px -2px rgba(16,24,40,.06);
    --radius:    12px;
    --radius-s:  8px;
}

html, body, .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text1) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow-s) !important;
}

[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSlider label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid white !important;
    box-shadow: var(--shadow-s) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-s) !important;
    font-size: 0.875rem !important;
    box-shadow: var(--shadow-s) !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: var(--radius-s) !important;
    padding: 0.75rem 1.25rem !important;
    box-shadow: 0 1px 2px rgba(21,112,239,.3) !important;
    transition: all .15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1762D6 !important;
    box-shadow: 0 4px 12px rgba(21,112,239,.35) !important;
    transform: translateY(-1px) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem !important;
    box-shadow: var(--shadow-s);
}
[data-testid="stMetricLabel"] > div {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: var(--text3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: var(--text1) !important;
    line-height: 1.2 !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2rem 0 !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: var(--text1) !important;
    letter-spacing: -0.02em !important;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow-s);
}
.card-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}

.insight {
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    line-height: 1.75;
}
.insight.high   { background:var(--danger-l); border:1px solid #FECDCA; }
.insight.medium { background:var(--warn-l);   border:1px solid #FEDF89; }
.insight.low    { background:var(--success-l);border:1px solid #A9EFC5; }
.insight-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}
.insight.high   .insight-title { color: #B42318; }
.insight.medium .insight-title { color: #B54708; }
.insight.low    .insight-title { color: #027A48; }
.insight-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
}
.insight-section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}
.insight.high   .insight-section-label { color:#B42318; }
.insight.medium .insight-section-label { color:#B54708; }
.insight.low    .insight-section-label { color:#027A48; }
.insight ul { margin:0; padding-left:1.1rem; color:var(--text2); }
.insight li  { margin-bottom:0.3rem; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text1);
    letter-spacing: -0.01em;
    margin-bottom: 1rem;
}

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
model = joblib.load("model/fraud_model.pkl")

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 1.25rem 1rem;">
        <div style="display:flex;align-items:center;gap:.65rem;
                    padding-bottom:1.25rem;border-bottom:1px solid #E4E7EC;margin-bottom:1.5rem;">
            <div style="width:34px;height:34px;background:#1570EF;border-radius:9px;
                        display:flex;align-items:center;justify-content:center;font-size:1rem;">🛡️</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.95rem;color:#101828;">FraudGuard</div>
                <div style="font-size:.68rem;color:#98A2B3;font-family:'DM Mono',monospace;letter-spacing:.05em;">AI Detection v2</div>
            </div>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:.68rem;font-weight:700;
                    color:#98A2B3;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.75rem;">
            Return Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 1.25rem 1.5rem;">', unsafe_allow_html=True)
        returns    = st.slider("Returns in last 90 days",        0,    15,    2)
        refund     = st.slider("Refund amount ($)",              100,  8000,  1200)
        gap        = st.slider("Purchase → Return gap (days)",   1,    30,    5)
        damage     = st.slider("Damage claim frequency",         0,    10,    1)
        high_value = st.selectbox("High value item?",            ["No", "Yes"])
        similarity = st.slider("Return reason similarity",       0.0,  1.0,   0.2)
        ratio      = st.slider("Return / Purchase ratio",        0.0,  1.0,   0.3)
        st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
        analyze = st.button("⚡  Run Analysis")
        st.markdown("</div></div>", unsafe_allow_html=True)

high_value_num = 1 if high_value == "Yes" else 0

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:flex-start;
            padding-bottom:1.5rem;border-bottom:1px solid #E4E7EC;margin-bottom:2rem;">
    <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                    color:#101828;letter-spacing:-.025em;line-height:1.15;">
            Return Fraud Detection
        </div>
        <div style="font-size:.875rem;color:#667085;margin-top:.35rem;font-family:'DM Sans',sans-serif;">
            AI-powered real-time fraud risk assessment for e-commerce returns
        </div>
    </div>
    <div style="font-family:'DM Mono',sans-serif;font-size:.72rem;color:#98A2B3;
                background:#F2F4F7;border:1px solid #E4E7EC;border-radius:6px;
                padding:.4rem .8rem;white-space:nowrap;margin-top:.25rem;">
        7 Feature Model
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
risk = None
if analyze:
    features = np.array([returns, refund, gap, damage,
                         high_value_num, similarity, ratio]).reshape(1, -1)
    risk = model.predict_proba(features)[0][1] * 100

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if risk is not None:

    if risk > 70:
        tier, cls, icon = "HIGH RISK",    "high",   "🔴"
        a_color, fill  = "#F04438",       "#FEF3F2"
        desc = "Immediate manual review recommended — significant fraud indicators detected."
    elif risk > 40:
        tier, cls, icon = "MEDIUM RISK",  "medium", "🟡"
        a_color, fill  = "#F79009",       "#FFFAEB"
        desc = "Moderate anomaly — verify customer history before approving refund."
    else:
        tier, cls, icon = "LOW RISK",     "low",    "🟢"
        a_color, fill  = "#12B76A",       "#ECFDF3"
        desc = "Behavior aligns with legitimate return patterns. Safe to proceed."

    # ── ROW 1: KPI CARDS ──────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Fraud Risk Score",    f"{risk:.1f}%")
    with c2: st.metric("Risk Tier",           tier.split()[0])
    with c3: st.metric("Returns (90 days)",   returns)
    with c4: st.metric("Refund Amount",        f"${refund:,}")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── ROW 2: GAUGE + STATUS ─────────────────
    g_col, s_col = st.columns([3, 2])

    with g_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Fraud Probability Score</div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%", "font": {"family": "Syne", "size": 52, "color": a_color}},
            gauge={
                "axis": {
                    "range": [0, 100], "tickwidth": 1,
                    "tickcolor": "#D0D5DD", "nticks": 6,
                    "tickfont": {"family": "DM Mono", "size": 10, "color": "#98A2B3"},
                },
                "bar": {"color": a_color, "thickness": 0.26},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "#F0FDF4"},
                    {"range": [40, 70],  "color": "#FFFBEB"},
                    {"range": [70, 100], "color": "#FFF1F0"},
                ],
                "threshold": {"line": {"color": a_color, "width": 3}, "thickness": 0.8, "value": risk},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(t=10, b=0, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with s_col:
        st.markdown(f"""
        <div class="card" style="height:100%;display:flex;flex-direction:column;gap:1rem;">
            <div class="card-label">Detection Status</div>
            <div style="flex:1;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;background:{fill};border-radius:8px;
                        padding:1.5rem;gap:.65rem;text-align:center;">
                <div style="font-size:2.75rem;line-height:1;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                            color:{a_color};letter-spacing:-.01em;">{tier}</div>
                <div style="font-size:.82rem;color:#475467;line-height:1.6;max-width:220px;">{desc}</div>
            </div>
            <div style="font-family:'DM Mono',sans-serif;font-size:.72rem;color:var(--text3);
                        text-align:center;">Score: {risk:.2f}% &nbsp;|&nbsp; {tier}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # ── ROW 3: CHARTS ─────────────────────────
    st.markdown('<div class="section-title">Behavioral Signal Analysis</div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)

    # Chart 1 — Horizontal feature bars
    with ch1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Input Feature Signals (Normalized 0–100%)</div>', unsafe_allow_html=True)

        labels   = ["Returns (90d)", "Refund Amount", "Return Gap", "Damage Claims",
                    "High Value Item", "Reason Similarity", "Return Ratio"]
        raw      = [returns, refund, gap, damage, high_value_num, similarity, ratio]
        maxes    = [15, 8000, 30, 10, 1, 1.0, 1.0]
        norms    = [round((v / m) * 100, 1) for v, m in zip(raw, maxes)]
        bar_cols = ["#F04438" if n > 70 else "#F79009" if n > 40 else "#1570EF" for n in norms]

        fig_bar = go.Figure(go.Bar(
            y=labels, x=norms, orientation="h",
            marker=dict(color=bar_cols, opacity=0.85, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in norms],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#667085"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=270, margin=dict(t=5, b=5, l=10, r=55),
            xaxis=dict(range=[0, 125], showgrid=True, gridcolor="#F2F4F7",
                       zeroline=False, showline=False,
                       ticksuffix="%", tickfont=dict(family="DM Mono", size=9, color="#98A2B3")),
            yaxis=dict(showgrid=False, showline=False,
                       tickfont=dict(family="DM Sans", size=11, color="#344054")),
            bargap=0.38,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart 2 — Scatter
    with ch2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Refund vs Damage Claims — Population View</div>', unsafe_allow_html=True)

        try:
            data = pd.read_csv("data/returns.csv")
            fig_sc = px.scatter(
                data, x="avg_refund_amount", y="damage_claim_frequency",
                color="fraud_label",
                color_discrete_map={0: "#1570EF", 1: "#F04438"},
                labels={"avg_refund_amount": "Avg Refund ($)",
                        "damage_claim_frequency": "Damage Claims",
                        "fraud_label": "Fraud"},
                opacity=0.45,
            )
            fig_sc.add_trace(go.Scatter(
                x=[refund], y=[damage], mode="markers", name="This Return",
                marker=dict(size=13, color=a_color,
                            line=dict(color="white", width=2.5), symbol="star"),
            ))
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=270, margin=dict(t=5, b=45, l=55, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.42,
                            xanchor="left", x=0,
                            font=dict(family="DM Sans", size=11, color="#667085"),
                            bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(showgrid=True, gridcolor="#F2F4F7", zeroline=False, showline=False,
                           tickfont=dict(family="DM Mono", size=9, color="#98A2B3"),
                           title=dict(text="Avg Refund ($)", font=dict(family="DM Sans", size=11, color="#667085"))),
                yaxis=dict(showgrid=True, gridcolor="#F2F4F7", zeroline=False, showline=False,
                           tickfont=dict(family="DM Mono", size=9, color="#98A2B3"),
                           title=dict(text="Damage Claims", font=dict(family="DM Sans", size=11, color="#667085"))),
            )
            st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.info("Add data/returns.csv to see population scatter.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart 3 — Histogram (full width)
    try:
        st.markdown("<div style='margin-top:1.25rem;'>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Return Frequency Distribution — All Customers</div>', unsafe_allow_html=True)

        fig_hist = px.histogram(data, x="returns_last_90_days", nbins=15,
                                color_discrete_sequence=["#1570EF"])
        fig_hist.add_vline(x=returns, line_color=a_color, line_width=2, line_dash="dash",
                           annotation_text=f"  This customer: {returns}",
                           annotation_font=dict(family="DM Mono", size=11, color=a_color),
                           annotation_position="top right")
        fig_hist.update_traces(marker=dict(opacity=0.72, line=dict(width=0)))
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=190, margin=dict(t=5, b=40, l=55, r=20), showlegend=False,
            bargap=0.07,
            xaxis=dict(showgrid=False, zeroline=False, showline=False,
                       tickfont=dict(family="DM Mono", size=9, color="#98A2B3"),
                       title=dict(text="Returns in last 90 days",
                                  font=dict(family="DM Sans", size=11, color="#667085"))),
            yaxis=dict(showgrid=True, gridcolor="#F2F4F7", zeroline=False, showline=False,
                       tickfont=dict(family="DM Mono", size=9, color="#98A2B3"),
                       title=dict(text="Count", font=dict(family="DM Sans", size=11, color="#667085"))),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div></div>", unsafe_allow_html=True)
    except Exception:
        pass

    st.divider()

    # ── ROW 4: INSIGHTS ───────────────────────
    st.markdown('<div class="section-title">Analysis & Recommended Actions</div>', unsafe_allow_html=True)

    if risk > 70:
        st.markdown(f"""
        <div class="insight high">
            <div class="insight-title">🚨 High Fraud Risk — Score: {risk:.1f}%</div>
            <div class="insight-grid">
                <div>
                    <div class="insight-section-label">Fraud Indicators</div>
                    <ul>
                        <li>Unusually high return frequency for this window</li>
                        <li>Multiple damage claims in short period</li>
                        <li>Refund amount exceeds typical customer baseline</li>
                        <li>Return reasons suggest templated/copy-paste submissions</li>
                    </ul>
                </div>
                <div>
                    <div class="insight-section-label">Recommended Actions</div>
                    <ul>
                        <li>Escalate to fraud analyst for manual review</li>
                        <li>Place refund on hold pending investigation</li>
                        <li>Flag account for enhanced future monitoring</li>
                        <li>Request proof-of-purchase documentation</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif risk > 40:
        st.markdown(f"""
        <div class="insight medium">
            <div class="insight-title">⚠️ Moderate Anomaly Detected — Score: {risk:.1f}%</div>
            <div class="insight-grid">
                <div>
                    <div class="insight-section-label">What We Found</div>
                    <ul>
                        <li>Return behavior deviates from customer baseline</li>
                        <li>Some indicators overlap with known fraud patterns</li>
                    </ul>
                </div>
                <div>
                    <div class="insight-section-label">Recommended Actions</div>
                    <ul>
                        <li>Review customer's full return history before approving</li>
                        <li>Request additional verification for high-value items</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="insight low">
            <div class="insight-title">✅ Low Risk — Likely Legitimate — Score: {risk:.1f}%</div>
            <ul>
                <li>Return behavior aligns with historical legitimate patterns for this customer segment</li>
                <li>No significant anomaly indicators detected across all 7 risk features</li>
                <li><strong>Recommendation:</strong> Proceed with standard refund processing workflow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  IDLE STATE
# ─────────────────────────────────────────────
else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:55vh;gap:1rem;text-align:center;">
        <div style="font-size:3.5rem;opacity:.2;">🛡️</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
                    color:#344054;">No Analysis Running</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:.875rem;color:#98A2B3;
                    max-width:280px;line-height:1.65;">
            Configure the return parameters in the sidebar, then click
            <strong style="color:#1570EF;">Run Analysis</strong> to see fraud detection results.
        </div>
    </div>
    """, unsafe_allow_html=True)