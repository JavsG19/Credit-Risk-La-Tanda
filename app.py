"""
Streamlit Web Application: CreditIQ Analytics — Motor de Riesgo y Aprobación de Crédito
Executive Fintech Underwriting & Portfolio Optimization Suite
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

from predictor import CreditRiskEngine

st.set_page_config(
    page_title="CreditIQ Analytics | Motor de Riesgo de Crédito",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Executive Fintech CSS (Zero Emojis, Inter + Tabular Nums)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    header[data-testid="stHeader"] {background: transparent !important;}

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px;
    }

    .tabular-nums, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], table, td, th {
        font-variant-numeric: tabular-nums !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 12px;
    }
    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: linear-gradient(135deg, #0F172A 0%, #0284C7 100%);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: center;
        letter-spacing: -0.03em;
    }
    .sidebar-brand-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.73rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 2px;
    }
    .sidebar-section-hdr {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-top: 1.1rem;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 4px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-bottom: 3px solid #0284C7;
        border-radius: 10px;
        padding: 22px 28px;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .hero-eyebrow {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #38BDF8;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin: 0 0 4px 0;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 0.88rem;
        font-weight: 400;
        color: #94A3B8;
        margin: 0;
        font-variant-numeric: tabular-nums;
    }
    .hero-chips {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
    }
    .hero-chip {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #E2E8F0;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-variant-numeric: tabular-nums;
    }
    .dot-green { width: 7px; height: 7px; border-radius: 50%; background-color: #10B981; display: inline-block; }
    .dot-amber { width: 7px; height: 7px; border-radius: 50%; background-color: #F59E0B; display: inline-block; }
    .dot-red   { width: 7px; height: 7px; border-radius: 50%; background-color: #EF4444; display: inline-block; }
    .dot-blue  { width: 7px; height: 7px; border-radius: 50%; background-color: #38BDF8; display: inline-block; }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        height: 100%;
    }
    .kpi-accent-slate   { border-top: 3px solid #0F172A; }
    .kpi-accent-sky     { border-top: 3px solid #0284C7; }
    .kpi-accent-emerald { border-top: 3px solid #059669; }
    .kpi-accent-crimson { border-top: 3px solid #DC2626; }

    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        margin-bottom: 8px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub b { color: #0F172A; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent !important;
        border: none !important;
        color: #64748B !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 0 4px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0284C7 !important;
        border-bottom: 2px solid #0284C7 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0284C7 !important;
    }

    .section-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
    }
    .section-desc {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .card-subtitle {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.7rem;
    }

    /* Verdict Cards */
    .verdict-card {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .verdict-low  { border-left: 4px solid #059669; background-color: #F0FDF4; }
    .verdict-mod  { border-left: 4px solid #D97706; background-color: #FFFBEB; }
    .verdict-high { border-left: 4px solid #DC2626; background-color: #FEF2F2; }

    .verdict-tag {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
    }
    .verdict-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .verdict-sub {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Benchmark HTML Table */
    .bench-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        font-variant-numeric: tabular-nums;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1.25rem;
    }
    .bench-table th {
        background-color: #F8FAFC;
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #E2E8F0;
    }
    .bench-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #F1F5F9;
        color: #334155;
    }
    .bench-table tr.winner-row {
        background-color: #F0F9FF;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-selected {
        background-color: #0284C7;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 8px;
        border-radius: 4px;
        margin-left: 8px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_credit_engine():
    models_dir = os.path.join(current_dir, "models")
    clean_csv = os.path.join(current_dir, "data", "processed", "credit_risk_clean.csv")
    return CreditRiskEngine(models_dir, clean_csv)


engine = load_credit_engine()

# -------------------------------------------------------------------------
# SIDEBAR: CONTROLES DEL SOLICITANTE Y MODELO
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-mark">IQ</div>
        <div>
            <p class="sidebar-brand-title">CreditIQ Analytics</p>
            <p class="sidebar-brand-sub">Fintech Underwriting &amp; Risk Engine</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-hdr">Casos de Ejemplo Rápidos</div>', unsafe_allow_html=True)
    preset = st.selectbox(
        "Cargar perfil predefinido",
        [
            "1. PyME / Negocio Sólido (Riesgo Bajo)",
            "2. Caso Frontera (Riesgo Moderado)",
            "3. Solicitante Sobre-Endeudado (Riesgo Alto)",
        ],
        index=0,
    )

    if preset.startswith("1."):
        def_income, def_amnt, def_rate = 78000, 12000, 10.5
        def_home_idx, def_intent_idx, def_grade_idx = 1, 0, 1  # MORTGAGE, VENTURE, B
        def_emp, def_age, def_hist, def_cb_idx = 6, 31, 7, 0
    elif preset.startswith("2."):
        def_income, def_amnt, def_rate = 42000, 14000, 14.2
        def_home_idx, def_intent_idx, def_grade_idx = 0, 1, 2  # RENT, DEBTCONSOLIDATION, C
        def_emp, def_age, def_hist, def_cb_idx = 2, 26, 4, 0
    else:
        def_income, def_amnt, def_rate = 28000, 16500, 17.8
        def_home_idx, def_intent_idx, def_grade_idx = 0, 1, 4  # RENT, DEBTCONSOLIDATION, E
        def_emp, def_age, def_hist, def_cb_idx = 1, 23, 2, 1

    st.markdown('<div class="sidebar-section-hdr">1. Perfil del Solicitante</div>', unsafe_allow_html=True)
    person_income = st.number_input("Ingreso Anual ($ USD)", min_value=4000, max_value=500000, value=def_income, step=2000)

    home_map = {
        "Rentada (RENT)": "RENT",
        "Hipotecada (MORTGAGE)": "MORTGAGE",
        "Propia (OWN)": "OWN",
        "Otra (OTHER)": "OTHER",
    }
    home_label = st.selectbox("Situación de Vivienda", list(home_map.keys()), index=def_home_idx)
    person_home_ownership = home_map[home_label]

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        person_age = st.slider("Edad (Años)", 20, 75, def_age)
    with c_p2:
        person_emp_length = st.slider("Antigüedad (Años)", 0, 30, def_emp)

    st.markdown('<div class="sidebar-section-hdr">2. Estructura del Crédito</div>', unsafe_allow_html=True)
    loan_amnt = st.number_input("Monto Solicitado ($ USD)", min_value=500, max_value=50000, value=def_amnt, step=500)
    loan_int_rate = st.slider("Tasa de Interés Anual (%)", 5.0, 24.0, float(def_rate), step=0.2)

    intent_map = {
        "Negocio / Capital de Trabajo (VENTURE)": "VENTURE",
        "Consolidación de Deudas (DEBTCONSOLIDATION)": "DEBTCONSOLIDATION",
        "Educación (EDUCATION)": "EDUCATION",
        "Gastos Médicos (MEDICAL)": "MEDICAL",
        "Mejoras del Hogar (HOMEIMPROVEMENT)": "HOMEIMPROVEMENT",
        "Gasto Personal (PERSONAL)": "PERSONAL",
    }
    intent_label = st.selectbox("Destino del Préstamo", list(intent_map.keys()), index=def_intent_idx)
    loan_intent = intent_map[intent_label]

    st.markdown('<div class="sidebar-section-hdr">3. Buró de Crédito y Motor</div>', unsafe_allow_html=True)
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        loan_grade = st.selectbox("Grado en Buró", ["A", "B", "C", "D", "E", "F", "G"], index=def_grade_idx)
    with c_b2:
        cb_label = st.selectbox("Impago Previo", ["No (Limpio)", "Sí (Con atraso)"], index=def_cb_idx)
        cb_person_default_on_file = "Y" if "Sí" in cb_label else "N"

    cb_person_cred_hist_length = st.slider("Años de Historial en Buró", 1, 25, def_hist)

    active_model = st.radio(
        "Modelo Activo de Decisión",
        ["Random Forest (300 Árboles)", "Regresión Logística (WoE Scorecard)"],
        index=0,
    )

# Run evaluation
res = engine.evaluate_applicant(
    person_age=person_age,
    person_income=float(person_income),
    person_home_ownership=person_home_ownership,
    person_emp_length=float(person_emp_length),
    loan_intent=loan_intent,
    loan_grade=loan_grade,
    loan_amnt=float(loan_amnt),
    loan_int_rate=float(loan_int_rate),
    cb_person_default_on_file=cb_person_default_on_file,
    cb_person_cred_hist_length=cb_person_cred_hist_length,
    lgd=0.75,
    active_model=active_model,
)

# -------------------------------------------------------------------------
# HERO BANNER & TOP 4 KPI CARDS
# -------------------------------------------------------------------------
if res["verdict_level"] == "low":
    dot_cls = "dot-green"
elif res["verdict_level"] == "mod":
    dot_cls = "dot-amber"
else:
    dot_cls = "dot-red"

st.markdown(f"""
<div class="hero-banner">
    <div>
        <div class="hero-eyebrow">MOTOR DE SUSCRIPCIÓN Y RIESGO CREDITICIO · FINTECH ANALYTICS</div>
        <h1 class="hero-title">Solicitud por ${loan_amnt:,.0f} USD · {intent_label.split(' (')[0]}</h1>
        <p class="hero-sub">Ingreso Anual: ${person_income:,.0f} USD · Ratio Deuda/Ingreso: {res['loan_percent_income']*100:.1f}% · Tasa: {loan_int_rate:.1f}% · Grado Buró: {loan_grade}</p>
    </div>
    <div class="hero-chips">
        <div class="hero-chip"><span class="{dot_cls}"></span>Clasificación: <b>{res['tier']}</b></div>
        <div class="hero-chip"><span class="dot-blue"></span>Motor Activo: <b>{active_model}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card kpi-accent-slate">
        <div class="kpi-label">Score Crediticio (300–850)</div>
        <div class="kpi-value">{res['score_selected']} pts</div>
        <div class="kpi-sub">RF: <b>{res['score_rf']}</b> · Logística WoE: <b>{res['score_lr']}</b></div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card kpi-accent-sky">
        <div class="kpi-label">Probabilidad de Impago (PD)</div>
        <div class="kpi-value">{res['pd_selected']*100:.1f}%</div>
        <div class="kpi-sub">RF: <b>{res['pd_rf']*100:.1f}%</b> · Logística WoE: <b>{res['pd_lr']*100:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    pnl_sign = "+" if res["expected_profit"] >= 0 else "-"
    pnl_accent = "kpi-accent-emerald" if res["expected_profit"] >= 0 else "kpi-accent-crimson"
    st.markdown(f"""
    <div class="kpi-card {pnl_accent}">
        <div class="kpi-label">Resultado Esperado del Crédito</div>
        <div class="kpi-value">{pnl_sign}${abs(res['expected_profit']):,.0f}</div>
        <div class="kpi-sub">Interés: <b>+${res['interest_revenue']:,.0f}</b> · Pérdida Esp. (EL): <b>-${res['expected_loss']:,.0f}</b></div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card kpi-accent-slate">
        <div class="kpi-label">Tasa Mínima de Equilibrio</div>
        <div class="kpi-value">{res['break_even_rate']:.1f}%</div>
        <div class="kpi-sub">Tasa ofrecida actual: <b>{loan_int_rate:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Dictamen de Solicitud y Explicabilidad",
    "Simulador de Cartera (Director de Riesgos)",
    "Validación de Modelos, Ranking IV y Curva KS",
])

# -------------------------------------------------------------------------
# TAB 1: DICTAMEN INDIVIDUAL Y EXPLICABILIDAD
# -------------------------------------------------------------------------
with tab1:
    if res["verdict_level"] == "low":
        v_cls = "verdict-low"
        v_color = "#059669"
        v_msg = (
            f"El solicitante presenta una probabilidad de impago baja ({res['pd_selected']*100:.1f}%) y la tasa de interés del {loan_int_rate:.1f}% "
            f"supera ampliamente la tasa mínima de equilibrio ({res['break_even_rate']:.1f}%). El crédito genera una utilidad neta esperada de +${res['expected_profit']:,.0f} USD."
        )
    elif res["verdict_level"] == "mod":
        v_cls = "verdict-mod"
        v_color = "#D97706"
        v_msg = (
            f"El perfil presenta riesgo moderado ({res['pd_selected']*100:.1f}% de probabilidad de impago). "
            f"Para que el préstamo sea rentable para la institución, se requiere una tasa de interés mínima de {res['break_even_rate']:.1f}% o reducir el monto solicitado."
        )
    else:
        v_cls = "verdict-high"
        v_color = "#DC2626"
        v_msg = (
            f"Riesgo elevado de incumplimiento ({res['pd_selected']*100:.1f}%). La Pérdida Esperada (-${res['expected_loss']:,.0f} USD) "
            f"supera los ingresos por intereses (+${res['interest_revenue']:,.0f} USD), arrojando un valor esperado negativo de -${abs(res['expected_profit']):,.0f} USD."
        )

    st.markdown(f"""
    <div class="verdict-card {v_cls}">
        <div class="verdict-tag" style="color: {v_color};">DICTAMEN AUTOMÁTICO DEL COMITÉ DE RIESGOS</div>
        <div class="verdict-title">{res['verdict']}</div>
        <div class="verdict-sub">{v_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1.25, 1])

    with col_e1:
        with st.container(border=True):
            st.markdown('<div class="card-subtitle">¿Qué factores subieron o bajaron su Score? (Explicabilidad WoE)</div>', unsafe_allow_html=True)
            df_imp = res["impacts_df"]
            bar_colors = ["#059669" if val >= 0 else "#DC2626" for val in df_imp["Impacto_Puntos"]]

            fig_imp = go.Figure(go.Bar(
                x=df_imp["Impacto_Puntos"],
                y=df_imp["Variable"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.1f} pts" for v in df_imp["Impacto_Puntos"]],
                textposition="outside",
                textfont=dict(family="Inter, sans-serif", size=11, color="#0F172A"),
                hovertemplate="<b>%{y}</b><br>Impacto en Score: %{x:+.1f} puntos<extra></extra>",
            ))
            max_abs = max(abs(df_imp["Impacto_Puntos"].min()), abs(df_imp["Impacto_Puntos"].max()), 20) * 1.32
            fig_imp.update_layout(
                font=dict(family="Inter, sans-serif", color="#0F172A"),
                xaxis_title="Puntos sumados (+) o restados (-) al Score base",
                xaxis=dict(range=[-max_abs, max_abs], zeroline=True, zerolinecolor="#94A3B8", showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=False),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=20, t=10, b=10),
                height=370,
            )
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
            st.caption("Barras verdes (+): Características que protegen al solicitante y suben su puntaje · Barras rojas (-): Factores que incrementan su riesgo.")

    with col_e2:
        with st.container(border=True):
            st.markdown('<div class="card-subtitle">anatomía Financiera del Préstamo ($ USD)</div>'.replace("anatomía", "Anatomía"), unsafe_allow_html=True)
            fin_cats = ["Interés Si Paga", "Pérdida Esperada (EL)", "Ganancia Neta Esperada"]
            fin_vals = [res["interest_revenue"], -res["expected_loss"], res["expected_profit"]]
            fin_cols = ["#0284C7", "#DC2626", "#059669" if res["expected_profit"] >= 0 else "#DC2626"]

            fig_fin = go.Figure(go.Bar(
                x=fin_cats,
                y=fin_vals,
                marker_color=fin_cols,
                text=[f"${v:+,.0f}" for v in fin_vals],
                textposition="outside",
                textfont=dict(family="Inter, sans-serif", size=12, color="#0F172A"),
            ))
            y_bound = max(abs(min(fin_vals)), abs(max(fin_vals)), 500) * 1.32
            fig_fin.update_layout(
                font=dict(family="Inter, sans-serif", color="#0F172A"),
                yaxis_title="Monto en Dólares ($ USD)",
                yaxis=dict(range=[-y_bound, y_bound], zeroline=True, zerolinecolor="#94A3B8", showgrid=True, gridcolor="#F1F5F9"),
                xaxis=dict(showgrid=False),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=20, b=10),
                height=265,
            )
            st.plotly_chart(fig_fin, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px; font-size: 0.83rem; color: #334155; line-height: 1.5; font-variant-numeric: tabular-nums;">
                <b>¿Cómo se calcula la Pérdida Esperada (EL)?</b><br>
                • <b>Monto expuesto (EAD):</b> ${loan_amnt:,.0f} USD<br>
                • <b>Severidad de pérdida si no paga (LGD):</b> 75% (${loan_amnt*0.75:,.0f} USD)<br>
                • <b>Pérdida Esperada (PD × LGD × EAD):</b> ${res['expected_loss']:,.0f} USD
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 2: SIMULADOR DE CARTERA (DIRECTOR DE RIESGOS)
# -------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Simulador de Política de Aprobación y Rentabilidad de Cartera</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Simulación financiera real sobre las <b>6,482 solicitudes de crédito del conjunto de prueba</b>. '
        'Elige hasta qué porcentaje máximo de riesgo (PD) estás dispuesto a aprobar y observa cómo cambia la ganancia total de la empresa.</div>',
        unsafe_allow_html=True,
    )

    sim_c1, sim_c2 = st.columns([1, 2.2])
    with sim_c1:
        with st.container(border=True):
            st.markdown('<div class="card-subtitle">Política de Corte del Banco</div>', unsafe_allow_html=True)
            cutoff_pct = st.slider(
                "Aprobar solo clientes con Probabilidad de Impago menor a:",
                min_value=5,
                max_value=100,
                value=22,
                step=1,
                format="%d%%",
            )
            lgd_pct = st.slider(
                "Severidad de Pérdida en Impago (LGD %):",
                min_value=40,
                max_value=100,
                value=75,
                step=5,
                format="%d%%",
                help="Porcentaje del préstamo que el banco pierde definitivamente cuando un cliente cae en impago (después de cobranza).",
            )
            sim = engine.simulate_portfolio_policy(
                max_pd_cutoff=cutoff_pct / 100.0,
                lgd=lgd_pct / 100.0,
                active_model=active_model,
            )
            st.markdown(f"""
            <div style="background-color: #F0F9FF; border-left: 3px solid #0284C7; padding: 12px 14px; border-radius: 4px; font-size: 0.84rem; color: #0F172A; margin-top: 8px; line-height: 1.55; font-variant-numeric: tabular-nums;">
                <b>Punto de Corte Óptimo Matemático:</b><br>
                Con este modelo, la ganancia máxima de la empresa se alcanza aprobando clientes con hasta <b>{sim['optimal_cutoff_pct']:.0f}%</b> de probabilidad de impago (Utilidad máxima: <b>+${sim['optimal_max_profit']:,.0f} USD</b>).
            </div>
            """, unsafe_allow_html=True)

    with sim_c2:
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Solicitudes Aprobadas", f"{sim['approval_rate']:.1f}%", f"{sim['n_approved']:,d} de {sim['total_apps']:,d}")
        with p2:
            st.metric("Morosidad de Cartera", f"{sim['port_default_rate']:.2f}%", f"{sim['port_default_rate'] - sim['base_default_rate']:+.2f}% vs sin filtro", delta_color="inverse")
        with p3:
            st.metric("Utilidad Neta Cartera", f"${sim['net_profit']:,.0f}", f"ROA: {sim['roi_pct']:+.2f}%")
        with p4:
            st.metric("Ahorro / Valor del Modelo", f"+${sim['value_added_vs_base']:,.0f}", f"Sin modelo: ${sim['base_net_profit']:,.0f}")

        with st.container(border=True):
            st.markdown('<div class="card-subtitle">Curva de Ganancia Neta de la Empresa ($) según el Punto de Corte</div>', unsafe_allow_html=True)
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=sim["curve"]["cutoff_pct"],
                y=sim["curve"]["net_profit"],
                mode="lines+markers",
                line=dict(color="#0284C7", width=3),
                marker=dict(size=5),
                name="Utilidad Neta ($ USD)",
                hovertemplate="<b>Corte PD ≤ %{x}%</b><br>Utilidad Neta: $%{y:,.0f} USD<extra></extra>",
            ))
            fig_curve.add_vline(
                x=cutoff_pct,
                line_width=2,
                line_dash="dash",
                line_color="#059669",
                annotation_text=f"Tu corte actual: {cutoff_pct}%",
                annotation_position="top left",
            )
            fig_curve.add_hline(y=0, line_width=1, line_color="#94A3B8")
            fig_curve.update_layout(
                font=dict(family="Inter, sans-serif", color="#0F172A"),
                xaxis_title="Umbral Máximo de Probabilidad de Impago Permitido (%)",
                yaxis_title="Utilidad Neta Total de la Cartera ($ USD)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=20, b=10),
                height=300,
            )
            st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False})

# -------------------------------------------------------------------------
# TAB 3: VALIDACIÓN DE MODELOS, RANKING IV Y CURVA KS
# -------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Comparativa de Modelos y Explicación Visual del Estadístico KS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Resultados evaluados fuera de muestra sobre <b>6,482 préstamos de prueba</b> (partición estratificada 80/20 sin fuga de datos).</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <table class="bench-table">
        <thead>
            <tr>
                <th>Arquitectura del Modelo</th>
                <th>ROC-AUC</th>
                <th>Índice Gini</th>
                <th>Estadístico KS</th>
                <th>Brier Score</th>
                <th>Exactitud (Acc)</th>
                <th>F1-Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1. Regresión Logística (WoE Scorecard Bancario)</td>
                <td>0.8746</td>
                <td>0.7493</td>
                <td>62.62%</td>
                <td>0.1018</td>
                <td>85.7%</td>
                <td>0.6846</td>
            </tr>
            <tr class="winner-row">
                <td>2. Random Forest Classifier (300 Árboles) <span class="badge-selected">Ganador</span></td>
                <td>0.9330</td>
                <td>0.8661</td>
                <td>72.98%</td>
                <td>0.0578</td>
                <td>93.0%</td>
                <td>0.8239</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    col_v1, col_v2 = st.columns([1.15, 1])

    with col_v1:
        with st.container(border=True):
            st.markdown('<div class="card-subtitle">¿Qué es el Estadístico KS? (Míralo en la gráfica)</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 0.85rem; color: #475569; margin-bottom: 10px; line-height: 1.5;">'
                'Si ordenamos a todos los clientes del más riesgoso al más seguro, la <b>línea roja</b> muestra qué porcentaje de todos los morosos vamos atrapando, '
                'y la <b>línea azul</b> cuántos buenos clientes dejamos fuera. <b>El KS (72.98%) es la distancia vertical máxima entre las dos curvas</b> (la línea punteada).'
                '</div>',
                unsafe_allow_html=True,
            )

            df_t = engine.test_portfolio.sort_values(by="pd_random_forest", ascending=False).reset_index(drop=True)
            df_t["pct_population"] = (np.arange(1, len(df_t) + 1) / len(df_t)) * 100.0
            df_t["cum_bads"] = (df_t["loan_status"].cumsum() / df_t["loan_status"].sum()) * 100.0
            df_t["cum_goods"] = (((1 - df_t["loan_status"]).cumsum()) / (1 - df_t["loan_status"]).sum()) * 100.0
            df_t["ks_gap"] = df_t["cum_bads"] - df_t["cum_goods"]

            max_ks_idx = int(df_t["ks_gap"].idxmax())
            ks_pop_x = float(df_t.loc[max_ks_idx, "pct_population"])
            ks_val_max = float(df_t.loc[max_ks_idx, "ks_gap"])

            # Downsample for fast rendering
            step_idx = np.linspace(0, len(df_t) - 1, 150).astype(int)
            df_plot = df_t.iloc[step_idx]

            fig_ks = go.Figure()
            fig_ks.add_trace(go.Scatter(
                x=df_plot["pct_population"], y=df_plot["cum_bads"],
                mode="lines", name="% Morosos Atrapados", line=dict(color="#DC2626", width=3)
            ))
            fig_ks.add_trace(go.Scatter(
                x=df_plot["pct_population"], y=df_plot["cum_goods"],
                mode="lines", name="% Buenos Pagadores", line=dict(color="#0284C7", width=3)
            ))
            fig_ks.add_vline(
                x=ks_pop_x, line_width=2, line_dash="dash", line_color="#0F172A",
                annotation_text=f"Máxima separación (KS = {ks_val_max:.1f}%)",
                annotation_position="top right"
            )
            fig_ks.update_layout(
                font=dict(family="Inter, sans-serif", color="#0F172A"),
                xaxis_title="% de Solicitudes Ordenadas de Mayor a Menor Riesgo",
                yaxis_title="% Acumulado Capturado",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=30, b=10), height=320
            )
            st.plotly_chart(fig_ks, use_container_width=True, config={"displayModeBar": False})

    with col_v2:
        with st.container(border=True):
            st.markdown('<div class="card-subtitle">Ranking de Variables por Information Value (IV)</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 0.85rem; color: #475569; margin-bottom: 10px; line-height: 1.5;">'
                'Mide qué variables tienen mayor poder para separar a un buen pagador de un moroso antes de entrenar el modelo.'
                '</div>',
                unsafe_allow_html=True,
            )
            df_iv_show = engine.iv_table[["Nombre_Negocio", "Information_Value_IV", "Poder_Predictivo"]].copy()
            df_iv_show.columns = ["Variable del Solicitante", "Valor IV", "Nivel Predictivo"]
            st.dataframe(
                df_iv_show,
                column_config={"Valor IV": st.column_config.NumberColumn("Valor IV", format="%.4f")},
                use_container_width=True,
                hide_index=True,
                height=330,
            )

st.markdown("""
<div style="border-top: 1px solid #E2E8F0; margin-top: 2.5rem; padding-top: 1.25rem; text-align: center; color: #94A3B8; font-size: 12px; font-weight: 500;">
    CreditIQ Analytics · Motor de Suscripción y Riesgo Crediticio · Desarrollado con Python, Scikit-Learn, Plotly y Streamlit
</div>
""", unsafe_allow_html=True)
