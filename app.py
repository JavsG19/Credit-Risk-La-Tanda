"""
Streamlit Web Application: La Tanda — Plataforma de Suscripción y Riesgo Crediticio
Warm Copper & Slate Executive Fintech Design · Bento Grid Architecture
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

import predictor
importlib.reload(predictor)
from predictor import CreditRiskEngine

st.set_page_config(
    page_title="La Tanda | Suscripción y Riesgo Crediticio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean mapping for feature labels without any parentheses
CLEAN_FEATURE_LABELS = {
    "Ratio Deuda / Ingreso (Loan-to-Income)": "Ratio Deuda a Ingreso",
    "Calificación Preliminar de Buró (Grado A-G)": "Grado en Buró",
    "Tasa de Interés Asignada (%)": "Tasa de Interés",
    "Ingreso Anual del Solicitante ($)": "Ingreso Anual",
    "Situación de Vivienda (Propia/Renta/Hipoteca)": "Tipo de Vivienda",
    "Impago Histórico en Buró (Sí/No)": "Historial de Impago",
    "Propósito / Destino del Crédito": "Destino del Crédito",
    "Monto Solicitado ($)": "Monto Solicitado",
    "Antigüedad Laboral / Negocio (Años)": "Antigüedad Laboral",
    "Edad del Solicitante": "Edad del Cliente",
    "Antigüedad en Buró de Crédito (Años)": "Antigüedad en Buró",
    "loan_percent_income": "Ratio Deuda a Ingreso",
    "loan_grade": "Grado en Buró",
    "loan_int_rate": "Tasa de Interés",
    "person_income": "Ingreso Anual",
    "person_home_ownership": "Tipo de Vivienda",
    "cb_person_default_on_file": "Historial de Impago",
    "loan_intent": "Destino del Crédito",
    "loan_amnt": "Monto Solicitado",
    "person_emp_length": "Antigüedad Laboral",
    "person_age": "Edad del Cliente",
    "cb_person_cred_hist_length": "Antigüedad en Buró",
}

def clean_label(raw_text: str) -> str:
    if raw_text in CLEAN_FEATURE_LABELS:
        return CLEAN_FEATURE_LABELS[raw_text]
    return raw_text.split("(")[0].strip()

# Global Warm Copper & Slate CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1C1917;
        background-color: #FAFAF9;
    }

    /* Hide Streamlit default chrome and left sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    header[data-testid="stHeader"] {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        max-width: 1360px !important;
    }

    /* Tabular numerals across all figures, metrics, and tables */
    .tabular-nums, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], table, td, th {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Slim Executive Top Header */
    .top-navbar {
        background-color: #1C1917;
        border-bottom: 2px solid #D97742;
        border-radius: 10px;
        padding: 14px 24px;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(28, 25, 23, 0.06);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 13px;
    }
    .nav-logo {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background-color: #D97742;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: center;
        letter-spacing: -0.02em;
    }
    .nav-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.15;
    }
    .nav-subtitle {
        font-size: 0.76rem;
        color: #A8A29E;
        font-weight: 500;
        margin: 2px 0 0 0;
    }
    .nav-badges {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .nav-pill {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(217, 119, 66, 0.35);
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.73rem;
        font-weight: 600;
        color: #F5F5F4;
        font-variant-numeric: tabular-nums;
    }

    /* Horizontal Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 1px solid #E7E5E4;
        margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent !important;
        border: none !important;
        color: #57534E !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        padding: 0 4px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1C1917 !important;
        border-bottom: 2px solid #D97742 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #D97742 !important;
    }

    /* Dossier Section Headers (Warm Copper Tint) */
    .dossier-hdr {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9A3412;
        background-color: #FDF3ED;
        border-left: 3px solid #D97742;
        padding: 6px 12px;
        border-radius: 0 6px 6px 0;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    /* Card Headers */
    .card-title-main {
        font-size: 1.02rem;
        font-weight: 700;
        color: #1C1917;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
    }
    .card-subtitle-muted {
        font-size: 0.8rem;
        color: #57534E;
        margin-bottom: 12px;
    }

    /* Bento Financial Ledger Grid */
    .ledger-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 8px;
    }
    .ledger-box {
        background-color: #FAFAF9;
        border: 1px solid #E7E5E4;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .ledger-lbl {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #57534E;
        margin-bottom: 4px;
    }
    .ledger-val {
        font-size: 1.32rem;
        font-weight: 800;
        color: #1C1917;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .ledger-note {
        font-size: 0.75rem;
        color: #57534E;
        margin-top: 4px;
        font-variant-numeric: tabular-nums;
    }

    /* Verdict Banner */
    .stamp-banner {
        background-color: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 3px rgba(28, 25, 23, 0.03);
    }
    .stamp-approved {
        border-left: 4px solid #15803D;
    }
    .stamp-cond {
        border-left: 4px solid #D97742;
    }
    .stamp-rejected {
        border-left: 4px solid #DC2626;
    }
    .stamp-badge {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 6px 13px;
        border-radius: 6px;
        font-variant-numeric: tabular-nums;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1C1917;
        letter-spacing: -0.01em;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .section-desc {
        font-size: 0.88rem;
        color: #57534E;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }

    /* Executive Benchmark Table */
    .fin-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        background: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 10px;
        overflow: hidden;
        font-variant-numeric: tabular-nums;
    }
    .fin-table th {
        background-color: #F5F5F4;
        color: #1C1917;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 14px 18px;
        text-align: left;
        border-bottom: 1px solid #E7E5E4;
    }
    .fin-table td {
        padding: 15px 18px;
        border-bottom: 1px solid #F5F5F4;
        color: #1C1917;
    }
    .fin-table tr.win-row {
        background-color: #FDF3ED;
        font-weight: 700;
    }
    .win-pill {
        background-color: #D97742;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 99px;
        margin-left: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Highlight Callout */
    .copper-callout {
        background-color: #FDF3ED;
        border-left: 3px solid #D97742;
        padding: 14px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #1C1917;
        line-height: 1.55;
        font-variant-numeric: tabular-nums;
    }

    /* Champion & Challenger Badges */
    .badge-champion {
        background-color: #E7E5E4;
        color: #1C1917;
        border: 1px solid #D6D3D1;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 99px;
        margin-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        vertical-align: middle;
    }
    .badge-challenger {
        background-color: #FDF3ED;
        color: #9A3412;
        border: 1px solid #FDBA74;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 99px;
        margin-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        vertical-align: middle;
    }

    /* A/B Impact Metric Cards */
    .ab-card {
        background-color: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(28, 25, 23, 0.03);
    }
    .ab-card-lbl {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #57534E;
        margin-bottom: 6px;
    }
    .ab-card-val {
        font-size: 1.48rem;
        font-weight: 800;
        color: #1C1917;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
        margin-bottom: 4px;
    }
    .ab-card-sub {
        font-size: 0.78rem;
        color: #57534E;
        font-variant-numeric: tabular-nums;
    }

    /* Governance Roadmap Cards */
    .gov-card {
        background-color: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-top: 3px solid #D97742;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(28, 25, 23, 0.03);
    }
    .gov-step-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1C1917;
        margin-bottom: 6px;
    }
    .gov-step-desc {
        font-size: 0.84rem;
        color: #57534E;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_credit_engine(cache_version: str = "2.0"):
    models_dir = os.path.join(current_dir, "models")
    clean_csv = os.path.join(current_dir, "data", "processed", "credit_risk_clean.csv")
    return CreditRiskEngine(models_dir, clean_csv)


engine = load_credit_engine(cache_version="2.0")

# -------------------------------------------------------------------------
# 3. ENCABEZADO Y CONTROLES SECUNDARIOS COMPACTOS
# -------------------------------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div class="nav-brand">
        <div class="nav-logo">LT</div>
        <div>
            <p class="nav-title">La Tanda · Plataforma de Suscripción y Riesgo Crediticio</p>
            <p class="nav-subtitle">Originación Financiera · Modelos de Calificación y Rentabilidad de Cartera</p>
        </div>
    </div>
    <div class="nav-badges">
        <span class="nav-pill">CARTERA AUDITADA: 32,409 CRÉDITOS</span>
        <span class="nav-pill">MUESTRA DE VALIDACIÓN: 6,482</span>
        <span class="nav-pill">ROC-AUC: 0.9330 · KS: 72.98%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Secondary Compact Controls (Zero Parentheses)
ctrl_c1, ctrl_c2 = st.columns([1.2, 1.2])
with ctrl_c1:
    preset = st.selectbox(
        "Expediente de Prueba",
        [
            "Expediente A · Empresa Consolidada · Riesgo Bajo",
            "Expediente B · Perfil Intermedio · Riesgo Moderado",
            "Expediente C · Nivel de Deuda Elevado · Riesgo Alto",
        ],
        index=0,
    )
with ctrl_c2:
    active_model_label = st.selectbox(
        "Motor de Calificación Activo",
        [
            "Random Forest · Ensamble de 300 Árboles",
            "Regresión Logística · Scorecard Bancario WoE",
        ],
        index=0,
    )
    active_model = "Random Forest" if "Random Forest" in active_model_label else "Regresión Logística"

if preset.startswith("Expediente A"):
    def_income, def_amnt, def_rate = 78000, 12000, 10.5
    def_home_idx, def_intent_idx, def_grade_idx = 1, 0, 1
    def_emp, def_age, def_hist, def_cb_idx = 6, 31, 7, 0
elif preset.startswith("Expediente B"):
    def_income, def_amnt, def_rate = 42000, 14000, 14.2
    def_home_idx, def_intent_idx, def_grade_idx = 0, 1, 2
    def_emp, def_age, def_hist, def_cb_idx = 2, 26, 4, 0
else:
    def_income, def_amnt, def_rate = 28000, 16500, 17.8
    def_home_idx, def_intent_idx, def_grade_idx = 0, 1, 4
    def_emp, def_age, def_hist, def_cb_idx = 1, 23, 2, 1

# Top Horizontal Tabs
tab_eval, tab_port, tab_audit = st.tabs([
    "Evaluación Individual",
    "Simulador de Cartera",
    "Auditoría de Modelos",
])

# =========================================================================
# PESTAÑA 1: EVALUACIÓN INDIVIDUAL
# =========================================================================
with tab_eval:
    left_dossier, right_cockpit = st.columns([1.02, 1.98], gap="medium")

    # ----- COLUMNA IZQUIERDA: EXPEDIENTE DE SOLICITUD -----
    with left_dossier:
        with st.container(border=True):
            st.markdown('<div class="card-title-main">Expediente de Solicitud</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle-muted">Ajusta los parámetros financieros del cliente para actualizar el dictamen.</div>', unsafe_allow_html=True)

            st.markdown('<div class="dossier-hdr">I. Capacidad Financiera y Demografía</div>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                person_income = st.number_input("Ingreso Anual · USD", min_value=4000, max_value=500000, value=def_income, step=2000)
            with f2:
                home_map = {
                    "Rentada": "RENT",
                    "Hipotecada": "MORTGAGE",
                    "Propia": "OWN",
                    "Otra": "OTHER",
                }
                home_label = st.selectbox("Tipo de Vivienda", list(home_map.keys()), index=def_home_idx)
                person_home_ownership = home_map[home_label]

            f3, f4 = st.columns(2)
            with f3:
                person_age = st.slider("Edad del Cliente · Años", 20, 75, def_age)
            with f4:
                person_emp_length = st.slider("Antigüedad Laboral · Años", 0, 30, def_emp)

            st.markdown('<div class="dossier-hdr">II. Estructura del Crédito</div>', unsafe_allow_html=True)
            f5, f6 = st.columns(2)
            with f5:
                loan_amnt = st.number_input("Monto Solicitado · USD", min_value=500, max_value=50000, value=def_amnt, step=500)
            with f6:
                loan_int_rate = st.number_input("Tasa de Interés Anual · %", min_value=5.0, max_value=24.0, value=float(def_rate), step=0.2, format="%.1f")

            intent_map = {
                "Capital de Trabajo y Negocio": "VENTURE",
                "Consolidación de Deuda": "DEBTCONSOLIDATION",
                "Educación Superior": "EDUCATION",
                "Gastos Médicos": "MEDICAL",
                "Mejoras del Hogar": "HOMEIMPROVEMENT",
                "Consumo Personal": "PERSONAL",
            }
            intent_label = st.selectbox("Destino del Financiamiento", list(intent_map.keys()), index=def_intent_idx)
            loan_intent = intent_map[intent_label]

            st.markdown('<div class="dossier-hdr">III. Historial en Buró de Crédito</div>', unsafe_allow_html=True)
            f7, f8, f9 = st.columns([1, 1.3, 1.1])
            with f7:
                loan_grade = st.selectbox("Grado en Buró", ["A", "B", "C", "D", "E", "F", "G"], index=def_grade_idx)
            with f8:
                cb_label = st.selectbox("Historial de Impago", ["Sin Atrasos", "Con Atraso Previo"], index=def_cb_idx)
                cb_person_default_on_file = "Y" if "Atraso" in cb_label else "N"
            with f9:
                cb_person_cred_hist_length = st.number_input("Años en Buró", min_value=1, max_value=30, value=def_hist, step=1)

    # Evaluate applicant
    res = engine.evaluate_applicant(
        person_age=int(person_age),
        person_income=float(person_income),
        person_home_ownership=person_home_ownership,
        person_emp_length=float(person_emp_length),
        loan_intent=loan_intent,
        loan_grade=loan_grade,
        loan_amnt=float(loan_amnt),
        loan_int_rate=float(loan_int_rate),
        cb_person_default_on_file=cb_person_default_on_file,
        cb_person_cred_hist_length=int(cb_person_cred_hist_length),
        lgd=0.75,
        active_model=active_model,
    )

    # ----- COLUMNA DERECHA: DICTAMEN, SCORE Y CASCADA -----
    with right_cockpit:
        if res["verdict_level"] == "low":
            s_cls = "stamp-approved"
            s_badge_style = "background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC;"
            s_title = "Crédito Aprobado"
            s_desc = f"Operación rentable con utilidad neta esperada de +${res['expected_profit']:,.0f} USD y probabilidad de impago controlada en {res['pd_selected']*100:.1f}%."
        elif res["verdict_level"] == "mod":
            s_cls = "stamp-cond"
            s_badge_style = "background-color: #FDF3ED; color: #9A3412; border: 1px solid #FDBA74;"
            s_title = "Aprobación Condicionada a Ajuste de Tasa o Monto"
            s_desc = f"Nivel de riesgo intermedio con {res['pd_selected']*100:.1f}% de probabilidad de impago. Requiere tasa mínima de {res['break_even_rate']:.1f}% para cubrir el riesgo."
        else:
            s_cls = "stamp-rejected"
            s_badge_style = "background-color: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5;"
            s_title = "Solicitud Rechazada por Nivel de Riesgo"
            s_desc = f"La pérdida esperada de -${res['expected_loss']:,.0f} USD supera el ingreso potencial por intereses de +${res['interest_revenue']:,.0f} USD."

        st.markdown(f"""
        <div class="stamp-banner {s_cls}">
            <div style="max-width: 78%;">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #57534E;">RESOLUCIÓN DEL COMITÉ DE CRÉDITO · CLASIFICACIÓN {res['tier']}</div>
                <div style="font-size: 1.18rem; font-weight: 800; color: #1C1917; margin-top: 3px;">{s_title}</div>
                <div style="font-size: 0.85rem; color: #57534E; margin-top: 3px;">{s_desc}</div>
            </div>
            <div class="stamp-badge" style="{s_badge_style}">SCORE {res['score_selected']} PTS</div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Left = Score Crediticio Gauge, Right = Resumen Financiero y de Riesgo
        bento_r1_c1, bento_r1_c2 = st.columns([1.05, 1.15], gap="medium")

        with bento_r1_c1:
            with st.container(border=True):
                st.markdown('<div class="card-title-main">Score Crediticio</div>', unsafe_allow_html=True)
                st.markdown('<div class="card-subtitle-muted">Escala de 300 a 850 puntos</div>', unsafe_allow_html=True)

                gauge_bar_color = "#15803D" if res["score_selected"] >= 630 else ("#D97742" if res["score_selected"] >= 555 else "#DC2626")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=res["score_selected"],
                    number={"font": {"family": "Plus Jakarta Sans, sans-serif", "size": 28, "color": "#1C1917"}, "suffix": " pts"},
                    delta={"reference": 600, "font": {"size": 15}, "increasing": {"color": "#15803D"}, "decreasing": {"color": "#DC2626"}},
                    gauge={
                        "axis": {
                            "range": [300, 850],
                            "tickwidth": 1,
                            "tickcolor": "#A8A29E",
                            "tickvals": [300, 450, 555, 630, 750, 850],
                            "tickfont": {"family": "Plus Jakarta Sans, sans-serif", "size": 11, "color": "#57534E"},
                        },
                        "bar": {"color": gauge_bar_color, "thickness": 0.32},
                        "bgcolor": "#F5F5F4",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [300, 555], "color": "#FEE2E2"},
                            {"range": [555, 630], "color": "#FDF3ED"},
                            {"range": [630, 850], "color": "#DCFCE7"},
                        ],
                        "threshold": {
                            "line": {"color": "#1C1917", "width": 3},
                            "thickness": 0.78,
                            "value": 600,
                        },
                    },
                ))
                # Generous left/right margins so 300 and 850 labels never clip
                fig_gauge.update_layout(
                    font={"family": "Plus Jakarta Sans, sans-serif", "color": "#1C1917"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=38, r=38, t=26, b=14),
                    height=225,
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Random Forest: {res['score_rf']} puntos · Scorecard WoE: {res['score_lr']} puntos · Referencia base: 600 puntos.")

        with bento_r1_c2:
            with st.container(border=True):
                st.markdown('<div class="card-title-main">Resumen Financiero y de Riesgo</div>', unsafe_allow_html=True)
                st.markdown('<div class="card-subtitle-muted">Indicadores económicos de la operación individual</div>', unsafe_allow_html=True)

                pnl_color = "#15803D" if res["expected_profit"] >= 0 else "#DC2626"
                pnl_sign = "+" if res["expected_profit"] >= 0 else "-"

                st.markdown(f"""
                <div class="ledger-grid">
                    <div class="ledger-box">
                        <div class="ledger-lbl">Probabilidad de Impago</div>
                        <div class="ledger-val">{res['pd_selected']*100:.1f}%</div>
                        <div class="ledger-note">RF: {res['pd_rf']*100:.1f}% · WoE: {res['pd_lr']*100:.1f}%</div>
                    </div>
                    <div class="ledger-box">
                        <div class="ledger-lbl">Ratio Deuda a Ingreso</div>
                        <div class="ledger-val">{res['loan_percent_income']*100:.1f}%</div>
                        <div class="ledger-note">${loan_amnt:,.0f} sobre ${person_income:,.0f} USD</div>
                    </div>
                    <div class="ledger-box">
                        <div class="ledger-lbl">Interés vs Pérdida Esperada</div>
                        <div class="ledger-val">+${res['interest_revenue']:,.0f}</div>
                        <div class="ledger-note" style="color: #DC2626;">Pérdida esperada: -${res['expected_loss']:,.0f}</div>
                    </div>
                    <div class="ledger-box">
                        <div class="ledger-lbl">Utilidad Neta Esperada</div>
                        <div class="ledger-val" style="color: {pnl_color};">{pnl_sign}${abs(res['expected_profit']):,.0f}</div>
                        <div class="ledger-note">Tasa de equilibrio: {res['break_even_rate']:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.caption("La pérdida esperada pondera probabilidad de incumplimiento, severidad de pérdida del 75% y capital expuesto.")

        # Row 2: Full-Width Score Waterfall Chart (Clean Labels, Zero Parentheses)
        with st.container(border=True):
            st.markdown('<div class="card-title-main">Desglose del Score</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle-muted">Contribución de cada variable al puntaje final</div>', unsafe_allow_html=True)

            df_imp = res["impacts_df"].copy()
            df_imp["Variable_Clean"] = df_imp["Variable"].apply(clean_label)
            df_imp["abs_pts"] = df_imp["Impacto_Puntos"].abs()
            df_top = df_imp.sort_values(by="abs_pts", ascending=False).head(6)

            sum_impacts = float(df_top["Impacto_Puntos"].sum())
            base_anchor = round(res["score_lr"] - sum_impacts, 1)

            wf_label_map = {
                "Ratio Deuda a Ingreso": "Deuda a<br>Ingreso",
                "Grado en Buró": "Grado en<br>Buró",
                "Tasa de Interés": "Tasa de<br>Interés",
                "Ingreso Anual": "Ingreso<br>Anual",
                "Tipo de Vivienda": "Tipo de<br>Vivienda",
                "Historial de Impago": "Historial<br>de Impago",
                "Destino del Crédito": "Destino del<br>Crédito",
                "Monto Solicitado": "Monto<br>Solicitado",
                "Antigüedad Laboral": "Antigüedad<br>Laboral",
                "Edad del Cliente": "Edad del<br>Cliente",
                "Antigüedad en Buró": "Antigüedad<br>en Buró",
            }
            wrapped_vars = [wf_label_map.get(v, v) for v in df_top["Variable_Clean"].tolist()]

            wf_x = ["Puntaje<br>Base"] + wrapped_vars + ["Score<br>Final"]
            wf_y = [base_anchor] + df_top["Impacto_Puntos"].tolist() + [0]
            wf_measure = ["absolute"] + ["relative"] * len(df_top) + ["total"]
            wf_text = [f"{base_anchor:.0f}"] + [f"{v:+.1f}" for v in df_top["Impacto_Puntos"]] + [f"{res['score_lr']} pts"]

            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=wf_measure,
                x=wf_x,
                y=wf_y,
                text=wf_text,
                textposition="outside",
                textfont={"family": "Plus Jakarta Sans, sans-serif", "size": 11, "color": "#1C1917"},
                connector={"line": {"color": "#D6D3D1", "width": 1, "dash": "dot"}},
                increasing={"marker": {"color": "#15803D"}},
                decreasing={"marker": {"color": "#DC2626"}},
                totals={"marker": {"color": "#1C1917"}},
            ))
            y_min_wf = min(base_anchor, res["score_lr"], 300) - 70
            y_max_wf = max(base_anchor, res["score_lr"], 750) + 85
            fig_wf.update_layout(
                font={"family": "Plus Jakarta Sans, sans-serif", "color": "#1C1917"},
                yaxis_title="Puntos de Score",
                yaxis=dict(range=[y_min_wf, y_max_wf], showgrid=True, gridcolor="#F5F5F4", zeroline=False),
                xaxis=dict(showgrid=False, tickangle=0, tickfont=dict(size=11, color="#57534E")),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=12, r=12, t=22, b=24),
                height=310,
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar": False})

# =========================================================================
# PESTAÑA 2: SIMULADOR DE CARTERA
# =========================================================================
with tab_port:
    st.markdown('<div class="section-title">Simulador de Política de Aprobación y Rentabilidad de Cartera</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Evaluación financiera sobre las <b>6,482 solicitudes de validación</b>. '
        'Define el umbral de riesgo permitido para optimizar el balance entre volumen de colocación, tasa de morosidad y utilidad neta.</div>',
        unsafe_allow_html=True,
    )

    pol_c1, pol_c2 = st.columns([1.02, 2.28], gap="medium")

    with pol_c1:
        with st.container(border=True):
            st.markdown('<div class="card-title-main">Política de Originación</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle-muted">Parámetros de corte del comité de crédito</div>', unsafe_allow_html=True)

            cutoff_pct = st.slider(
                "Umbral Máximo de Impago",
                min_value=5,
                max_value=100,
                value=22,
                step=1,
                format="%d%%",
            )
            lgd_pct = st.slider(
                "Severidad de Pérdida",
                min_value=40,
                max_value=100,
                value=75,
                step=5,
                format="%d%%",
            )
            sim = engine.simulate_portfolio_policy(
                max_pd_cutoff=cutoff_pct / 100.0,
                lgd=lgd_pct / 100.0,
                active_model=active_model,
            )

            st.markdown(f"""
            <div class="copper-callout" style="margin-top: 12px;">
                <b>Punto de Equilibrio Óptimo</b><br>
                La rentabilidad máxima de la cartera se alcanza con un umbral de <b>{sim['optimal_cutoff_pct']:.0f}%</b>, generando una utilidad neta de <b>+${sim['optimal_max_profit']:,.0f} USD</b>.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #FAFAF9; border: 1px solid #E7E5E4; padding: 14px 16px; border-radius: 8px; font-size: 0.83rem; color: #1C1917; margin-top: 12px; line-height: 1.65; font-variant-numeric: tabular-nums;">
                <b>Balance Financiero de Cartera · Umbral {cutoff_pct}%</b><br>
                • Capital Colocado: <b>${sim['capital_deployed']:,.0f} USD</b><br>
                • Ingresos por Intereses: <b style="color:#15803D;">+${sim['interest_earned']:,.0f} USD</b><br>
                • Pérdida por Cartera Vencida: <b style="color:#DC2626;">-${sim['principal_lost']:,.0f} USD</b><br>
                • Utilidad Neta Realizada: <b>${sim['net_profit']:,.0f} USD</b>
            </div>
            """, unsafe_allow_html=True)

    with pol_c2:
        def_diff = sim["port_default_rate"] - sim["base_default_rate"]
        net_sign = "+" if sim["net_profit"] > 0 else ("-" if sim["net_profit"] < 0 else "")
        net_color = "#15803D" if sim["net_profit"] >= 0 else "#DC2626"
        base_sign = "-" if sim["base_net_profit"] < 0 else ""
        val_sign = "+" if sim["value_added_vs_base"] >= 0 else "-"
        val_color = "#15803D" if sim["value_added_vs_base"] >= 0 else "#DC2626"

        m1, m2, m3, m4 = st.columns(4, gap="small")
        with m1:
            st.markdown(f"""
            <div class="ab-card" style="padding: 15px 16px;">
                <div class="ab-card-lbl">Tasa de Aprobación</div>
                <div class="ab-card-val" style="font-size: 1.38rem;">{sim['approval_rate']:.1f}%</div>
                <div class="ab-card-sub" style="color: #15803D; font-weight: 600;">{sim['n_approved']:,d} de {sim['total_apps']:,d}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="ab-card" style="padding: 15px 16px;">
                <div class="ab-card-lbl">Índice de Morosidad</div>
                <div class="ab-card-val" style="font-size: 1.38rem;">{sim['port_default_rate']:.2f}%</div>
                <div class="ab-card-sub" style="color: #15803D; font-weight: 600;">{def_diff:+.2f}% vs base</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="ab-card" style="padding: 15px 16px;">
                <div class="ab-card-lbl">Utilidad Neta Cartera</div>
                <div class="ab-card-val" style="font-size: 1.38rem; color: {net_color};">{net_sign}${abs(sim['net_profit']):,.0f}</div>
                <div class="ab-card-sub" style="color: #57534E; font-weight: 600;">ROA {sim['roi_pct']:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="ab-card" style="padding: 15px 16px;">
                <div class="ab-card-lbl">Valor Agregado Modelo</div>
                <div class="ab-card-val" style="font-size: 1.38rem; color: {val_color};">{val_sign}${abs(sim['value_added_vs_base']):,.0f}</div>
                <div class="ab-card-sub" style="color: #57534E; font-weight: 600;">Sin filtro {base_sign}${abs(sim['base_net_profit']):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="card-title-main">Curva de Rentabilidad Neta de la Cartera</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle-muted">Utilidad acumulada en dólares según el umbral máximo de impago permitido</div>', unsafe_allow_html=True)

            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=sim["curve"]["cutoff_pct"],
                y=sim["curve"]["net_profit"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(217, 119, 66, 0.14)",
                line=dict(color="#D97742", width=3),
                name="Utilidad Neta",
                hovertemplate="<b>Umbral ≤ %{x}%</b><br>Utilidad Neta: $%{y:,.0f} USD<extra></extra>",
            ))
            fig_area.add_vline(
                x=cutoff_pct,
                line_width=2,
                line_dash="dash",
                line_color="#1C1917",
                annotation_text=f"Umbral activo: {cutoff_pct}%",
                annotation_position="top left",
                annotation_font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#1C1917"),
            )
            fig_area.add_hline(y=0, line_width=1, line_color="#A8A29E")
            fig_area.update_layout(
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#1C1917"),
                xaxis_title="Umbral Máximo de Impago Permitido · %",
                yaxis_title="Utilidad Neta de la Cartera · USD",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F5F5F4", zeroline=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=12, r=12, t=20, b=10),
                height=330,
            )
            st.plotly_chart(fig_area, use_container_width=True, config={"displayModeBar": False})

# =========================================================================
# PESTAÑA 3: AUDITORÍA DE MODELOS
# =========================================================================
with tab_audit:
    st.markdown('<div class="section-title">Auditoría de Modelos y Capacidad Predictiva</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Validación estadística fuera de muestra sobre <b>6,482 operaciones crediticias</b> '
        'comparando el desempeño del modelo paramétrico tradicional frente al ensamble no lineal de árboles de decisión.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <table class="fin-table">
        <thead>
            <tr>
                <th>Arquitectura del Modelo</th>
                <th>ROC-AUC</th>
                <th>Índice Gini</th>
                <th>Estadístico KS</th>
                <th>Brier Score</th>
                <th>Exactitud</th>
                <th>F1-Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Regresión Logística · Scorecard Bancario WoE <span class="badge-champion">CHAMPION</span></td>
                <td>0.8746</td>
                <td>0.7493</td>
                <td>62.62%</td>
                <td>0.1018</td>
                <td>85.7%</td>
                <td>0.6846</td>
            </tr>
            <tr class="win-row">
                <td>Random Forest Classifier · Ensamble de 300 Árboles <span class="badge-challenger">CHALLENGER</span></td>
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

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    aud_c1, aud_c2 = st.columns([1.08, 1], gap="medium")

    with aud_c1:
        with st.container(border=True):
            ks_hdr_c1, ks_hdr_c2 = st.columns([2.2, 1])
            with ks_hdr_c1:
                st.markdown('<div class="card-title-main">Curva de Separación Kolmogorov-Smirnov</div>', unsafe_allow_html=True)
                st.markdown('<div class="card-subtitle-muted">Capacidad de discriminación entre cartera vigente y cartera vencida</div>', unsafe_allow_html=True)
            with ks_hdr_c2:
                st.markdown(
                    '<div style="background-color: #FDF3ED; border: 1px solid #FDBA74; border-radius: 6px; padding: 6px 12px; text-align: right;">'
                    '<div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #9A3412;">ESTADÍSTICO KS</div>'
                    '<div style="font-size: 1.15rem; font-weight: 800; color: #1C1917;" class="tabular-nums">72.98%</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="font-size: 0.82rem; color: #57534E; margin-bottom: 10px; line-height: 1.5;">'
                'Evalúa la distancia acumulada máxima entre la distribución de clientes en incumplimiento y clientes cumplidos al ordenar la cartera por nivel de riesgo estimado.'
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

            step_idx = np.linspace(0, len(df_t) - 1, 150).astype(int)
            df_plot = df_t.iloc[step_idx]

            fig_ks = go.Figure()
            fig_ks.add_trace(go.Scatter(
                x=df_plot["pct_population"], y=df_plot["cum_bads"],
                mode="lines", name="Cartera Vencida Acumulada", line=dict(color="#D97742", width=3)
            ))
            fig_ks.add_trace(go.Scatter(
                x=df_plot["pct_population"], y=df_plot["cum_goods"],
                mode="lines", name="Cartera Vigente Acumulada", line=dict(color="#1C1917", width=2.5)
            ))
            fig_ks.add_vline(
                x=ks_pop_x, line_width=1.8, line_dash="dash", line_color="#57534E",
                annotation_text=f"Máxima separación: {ks_val_max:.1f}%",
                annotation_position="top right",
                annotation_font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#1C1917"),
            )
            fig_ks.update_layout(
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#1C1917"),
                xaxis_title="Porcentaje de Cartera Ordenada por Riesgo · %",
                yaxis_title="Porcentaje Acumulado · %",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F5F5F4", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=28, b=10), height=300
            )
            st.plotly_chart(fig_ks, use_container_width=True, config={"displayModeBar": False})

    with aud_c2:
        with st.container(border=True):
            st.markdown('<div class="card-title-main">Poder Predictivo por Variable</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle-muted">Jerarquía de variables según su índice Information Value</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 0.82rem; color: #57534E; margin-bottom: 10px; line-height: 1.5;">'
                'Las variables que superan el umbral de 0.30 constituyen los indicadores primarios de solvencia en el modelo de originación.'
                '</div>',
                unsafe_allow_html=True,
            )

            df_iv_p = engine.iv_table.sort_values(by="Information_Value_IV", ascending=True).copy()
            df_iv_p["Clean_Name"] = df_iv_p["Variable"].apply(clean_label)
            iv_colors = ["#D97742" if v >= 0.30 else ("#1C1917" if v >= 0.10 else "#A8A29E") for v in df_iv_p["Information_Value_IV"]]

            fig_iv = go.Figure(go.Bar(
                x=df_iv_p["Information_Value_IV"],
                y=df_iv_p["Clean_Name"],
                orientation="h",
                marker_color=iv_colors,
                text=[f"{v:.3f}" for v in df_iv_p["Information_Value_IV"]],
                textposition="outside",
                textfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#1C1917"),
            ))
            fig_iv.add_vline(
                x=0.30, line_width=1.5, line_dash="dot", line_color="#57534E",
                annotation_text="Umbral Fuerte · 0.30",
                annotation_font=dict(family="Plus Jakarta Sans, sans-serif", size=10, color="#57534E"),
            )
            fig_iv.update_layout(
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#1C1917"),
                xaxis_title="Índice Information Value",
                xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#F5F5F4", zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=24, t=14, b=10), height=325
            )
            st.plotly_chart(fig_iv, use_container_width=True, config={"displayModeBar": False})

    # ---------------------------------------------------------------------
    # B) ESTRATEGIA DE DESPLIEGUE GRADUAL CHAMPION VS. CHALLENGER (A/B TEST)
    # ---------------------------------------------------------------------
    st.markdown('<hr style="border: none; border-top: 1px solid #E7E5E4; margin: 2.25rem 0 1.75rem 0;">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Estrategia de Despliegue Champion vs. Challenger</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Simulación de asignación controlada de tráfico para validar el modelo retador sin comprometer la cartera</div>',
        unsafe_allow_html=True,
    )

    ab_col1, ab_col2 = st.columns([1.05, 1.95], gap="large")

    with ab_col1:
        with st.container(border=True):
            st.markdown('<div style="padding: 4px 4px 6px 4px;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title-main">Asignación de Tráfico A/B</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle-muted">Control gradual de exposición en originación</div>', unsafe_allow_html=True)

            challenger_pct = st.slider(
                "Porcentaje de solicitudes asignadas al Challenger",
                min_value=0,
                max_value=100,
                value=20,
                step=5,
                format="%d%%",
            )
            champion_pct = 100 - challenger_pct

            st.markdown(f"""
            <div style="margin-top: 14px; background-color: #FAFAF9; border: 1px solid #E7E5E4; border-radius: 8px; padding: 14px 16px;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #1C1917; margin-bottom: 10px;" class="tabular-nums">
                    {champion_pct}% Champion &nbsp;·&nbsp; {challenger_pct}% Challenger
                </div>
                <div style="display: flex; width: 100%; height: 10px; border-radius: 99px; overflow: hidden; background-color: #E7E5E4;">
                    <div style="width: {champion_pct}%; background-color: #57534E; height: 100%;"></div>
                    <div style="width: {challenger_pct}%; background-color: #D97742; height: 100%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #57534E; margin-top: 8px; font-weight: 600;">
                    <span>Regresión Logística WoE</span>
                    <span>Random Forest</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with ab_col2:
        sim_champ = engine.simulate_portfolio_policy(max_pd_cutoff=0.22, lgd=0.75, active_model="Regresión Logística")
        sim_chall = engine.simulate_portfolio_policy(max_pd_cutoff=0.22, lgd=0.75, active_model="Random Forest")

        w_chall = challenger_pct / 100.0
        combined_profit = (1.0 - w_chall) * sim_champ["net_profit"] + w_chall * sim_chall["net_profit"]
        incremental_profit = combined_profit - sim_champ["net_profit"]
        exposure_count = int(round(sim_chall["total_apps"] * w_chall))
        inc_color = "#15803D" if incremental_profit > 0 else "#1C1917"
        inc_sign = "+" if incremental_profit > 0 else ""

        ab_m1, ab_m2, ab_m3 = st.columns(3, gap="medium")
        with ab_m1:
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-card-lbl">Utilidad Neta Combinada</div>
                <div class="ab-card-val">${combined_profit:,.0f} USD</div>
                <div class="ab-card-sub">Resultado ponderado bajo umbral 22%</div>
            </div>
            """, unsafe_allow_html=True)
        with ab_m2:
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-card-lbl">Utilidad Incremental</div>
                <div class="ab-card-val" style="color: {inc_color};">{inc_sign}${incremental_profit:,.0f} USD</div>
                <div class="ab-card-sub">Valor extra frente a 100% Champion</div>
            </div>
            """, unsafe_allow_html=True)
        with ab_m3:
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-card-lbl">Exposición en Prueba</div>
                <div class="ab-card-val">{exposure_count:,d} créditos</div>
                <div class="ab-card-sub">Evaluados por Challenger de {sim_chall['total_apps']:,d}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="copper-callout">
            <b>Lectura Ejecutiva del Piloto</b><br>
            Bajo un esquema operativo con umbral de impago del 22%, operar al 100% con el modelo titular Champion genera <b>${sim_champ['net_profit']:,.0f} USD</b> netos. 
            Al desviar el <b>{challenger_pct}%</b> del tráfico hacia el modelo Challenger se capturan <b>{inc_sign}${incremental_profit:,.0f} USD</b> adicionales sin exponer la totalidad de la cartera.
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # C) HOJA DE RUTA DE GOBERNANZA BANCARIA (3 PASOS)
    # ---------------------------------------------------------------------
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card-title-main" style="margin-bottom: 12px;">Hoja de Ruta de Gobernanza</div>', unsafe_allow_html=True)

    gov_c1, gov_c2, gov_c3 = st.columns(3, gap="medium")
    with gov_c1:
        st.markdown("""
        <div class="gov-card">
            <div class="gov-step-title">Paso 1 · Modo Sombra</div>
            <div class="gov-step-desc">El modelo Challenger evalúa solicitudes en paralelo sin afectar decisiones reales para auditar su estabilidad.</div>
        </div>
        """, unsafe_allow_html=True)
    with gov_c2:
        st.markdown("""
        <div class="gov-card">
            <div class="gov-step-title">Paso 2 · Piloto Controlado</div>
            <div class="gov-step-desc">Se asigna una fracción acotada del tráfico real mediante pruebas A/B para confirmar el valor financiero incremental.</div>
        </div>
        """, unsafe_allow_html=True)
    with gov_c3:
        st.markdown("""
        <div class="gov-card">
            <div class="gov-step-title">Paso 3 · Promoción a Champion</div>
            <div class="gov-step-desc">Tras validar consistencia en cosechas reales y cumplimiento regulatorio, el retador asume como nuevo modelo titular.</div>
        </div>
        """, unsafe_allow_html=True)
