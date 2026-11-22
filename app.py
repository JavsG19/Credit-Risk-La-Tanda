"""
Streamlit Web Application: CreditIQ — Private Banking & Fintech Underwriting Cockpit
Full-Screen Bento Grid Architecture (No Left Sidebar) · Forest Emerald & Warm Alabaster Theme
Features: Circular FICO Score Gauge (300-850), Score Waterfall Chart, & Portfolio Profit Area Simulator
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

from predictor import CreditRiskEngine

st.set_page_config(
    page_title="CreditIQ | Comité de Suscripción y Riesgo Crediticio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global Private Banking & Financial Terminal CSS (Plus Jakarta Sans + JetBrains Mono, No Sidebar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #111827;
        background-color: #F5F6F3;
    }

    /* Hide Streamlit chrome and completely disable left sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    header[data-testid="stHeader"] {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1360px !important;
    }

    /* Monospace financial numerals */
    .mono-num, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], table, td {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Top Banking Navbar */
    .top-navbar {
        background-color: #064E3B;
        border-bottom: 3px solid #10B981;
        border-radius: 12px;
        padding: 16px 24px;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(6, 78, 59, 0.12);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .nav-logo {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        background-color: #10B981;
        color: #064E3B;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .nav-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.15;
    }
    .nav-subtitle {
        font-size: 0.78rem;
        color: #A7F3D0;
        font-weight: 500;
        margin: 2px 0 0 0;
    }
    .nav-badges {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .nav-pill {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(167, 243, 208, 0.28);
        border-radius: 999px;
        padding: 5px 13px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #ECFDF5;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Dossier Section Headers (Left Panel) */
    .dossier-hdr {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #064E3B;
        background-color: #ECFDF5;
        border-left: 3px solid #059669;
        padding: 6px 10px;
        border-radius: 0 4px 4px 0;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    /* Bento Ledger Cards */
    .ledger-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 8px;
    }
    .ledger-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 11px 14px;
    }
    .ledger-lbl {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6B7280;
        margin-bottom: 3px;
    }
    .ledger-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.2;
    }
    .ledger-note {
        font-size: 0.72rem;
        color: #6B7280;
        margin-top: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Stamp Verdict Banner */
    .stamp-banner {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .stamp-approved {
        background-color: #ECFDF5;
        border: 1.5px solid #059669;
    }
    .stamp-cond {
        background-color: #FFFBEB;
        border: 1.5px solid #D97706;
    }
    .stamp-rejected {
        background-color: #FEF2F2;
        border: 1.5px solid #DC2626;
    }
    .stamp-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 4px;
        color: #FFFFFF;
    }

    /* Custom Segmented Radio Pills */
    div[role="radiogroup"] {
        gap: 8px;
    }

    /* Benchmark Table */
    .fin-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        overflow: hidden;
    }
    .fin-table th {
        background-color: #064E3B;
        color: #ECFDF5;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 11px 14px;
        text-align: left;
    }
    .fin-table td {
        padding: 11px 14px;
        border-bottom: 1px solid #F3F4F6;
        color: #1F2937;
        font-family: 'JetBrains Mono', monospace;
    }
    .fin-table tr.win-row {
        background-color: #ECFDF5;
        font-weight: 700;
        color: #064E3B;
    }
    .win-pill {
        background-color: #059669;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 99px;
        margin-left: 6px;
        text-transform: uppercase;
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
# TOP NAVIGATION HEADER (NO SIDEBAR)
# -------------------------------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div class="nav-brand">
        <div class="nav-logo">IQ</div>
        <div>
            <p class="nav-title">CreditIQ · Comité de Suscripción y Riesgo Crediticio</p>
            <p class="nav-subtitle">Plataforma de Originación Fintech · WoE Scorecard &amp; Random Forest Ensemble</p>
        </div>
    </div>
    <div class="nav-badges">
        <span class="nav-pill">CARTERA AUDITADA: 32,409 CRÉDITOS</span>
        <span class="nav-pill">TEST OUT-OF-SAMPLE: 6,482</span>
        <span class="nav-pill">ROC-AUC: 0.9330 · KS: 72.98%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Control Strip: Preset Selector & Active Model Switcher
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.4, 1.4, 1.2])
with ctrl_c1:
    preset = st.selectbox(
        "Cargar Expediente Predefinido de Prueba",
        [
            "Expediente A · PyME / Negocio Consolidado (Riesgo Bajo)",
            "Expediente B · Solicitante Caso Frontera (Riesgo Moderado)",
            "Expediente C · Solicitante Sobre-Endeudado (Riesgo Crítico)",
        ],
        index=0,
    )
with ctrl_c2:
    active_model = st.selectbox(
        "Motor Matemático de Decisión Activo",
        ["Random Forest (300 Árboles · No Lineal)", "Regresión Logística (WoE Scorecard Bancario)"],
        index=0,
    )
with ctrl_c3:
    view_mode = st.selectbox(
        "Vista de Cabina",
        [
            "1. Mesa de Suscripción (Evaluador Individual)",
            "2. Comité de Cartera (Simulador de Rentabilidad)",
            "3. Auditoría Estadística (Curva KS & Ranking IV)",
        ],
        index=0,
    )

if preset.startswith("Expediente A"):
    def_income, def_amnt, def_rate = 78000, 12000, 10.5
    def_home_idx, def_intent_idx, def_grade_idx = 1, 0, 1  # MORTGAGE, VENTURE, B
    def_emp, def_age, def_hist, def_cb_idx = 6, 31, 7, 0
elif preset.startswith("Expediente B"):
    def_income, def_amnt, def_rate = 42000, 14000, 14.2
    def_home_idx, def_intent_idx, def_grade_idx = 0, 1, 2  # RENT, DEBTCONSOLIDATION, C
    def_emp, def_age, def_hist, def_cb_idx = 2, 26, 4, 0
else:
    def_income, def_amnt, def_rate = 28000, 16500, 17.8
    def_home_idx, def_intent_idx, def_grade_idx = 0, 1, 4  # RENT, DEBTCONSOLIDATION, E
    def_emp, def_age, def_hist, def_cb_idx = 1, 23, 2, 1

# =========================================================================
# VISTA 1: MESA DE SUSCRIPCIÓN (BENTO GRID DE PANTALLA COMPLETA)
# =========================================================================
if view_mode.startswith("1."):
    left_dossier, right_cockpit = st.columns([1.05, 1.95], gap="medium")

    # ----- COLUMNA IZQUIERDA: FORMULARIO BANCARIO DEL EXPEDIENTE -----
    with left_dossier:
        with st.container(border=True):
            st.markdown("<div style='font-size: 1.02rem; font-weight: 800; color: #064E3B; margin-bottom: 4px;'>Expediente de Solicitud</div>", unsafe_allow_html=True)
            st.caption("Modifica cualquier parámetro del solicitante para recalcular el dictamen en tiempo real.")

            st.markdown('<div class="dossier-hdr">I. Capacidad Financiera y Demografía</div>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                person_income = st.number_input("Ingreso Anual ($ USD)", min_value=4000, max_value=500000, value=def_income, step=2000)
            with f2:
                home_map = {
                    "Rentada (RENT)": "RENT",
                    "Hipotecada (MORTGAGE)": "MORTGAGE",
                    "Propia (OWN)": "OWN",
                    "Otra (OTHER)": "OTHER",
                }
                home_label = st.selectbox("Tipo de Vivienda", list(home_map.keys()), index=def_home_idx)
                person_home_ownership = home_map[home_label]

            f3, f4 = st.columns(2)
            with f3:
                person_age = st.slider("Edad del Cliente", 20, 75, def_age)
            with f4:
                person_emp_length = st.slider("Antigüedad Laboral (Años)", 0, 30, def_emp)

            st.markdown('<div class="dossier-hdr">II. Estructura del Crédito Solicitado</div>', unsafe_allow_html=True)
            f5, f6 = st.columns(2)
            with f5:
                loan_amnt = st.number_input("Monto Solicitado ($ USD)", min_value=500, max_value=50000, value=def_amnt, step=500)
            with f6:
                loan_int_rate = st.number_input("Tasa Anual Asignada (%)", min_value=5.0, max_value=24.0, value=float(def_rate), step=0.2, format="%.1f")

            intent_map = {
                "Negocio / Capital de Trabajo": "VENTURE",
                "Consolidación de Deudas": "DEBTCONSOLIDATION",
                "Educación Universitaria": "EDUCATION",
                "Gastos Médicos": "MEDICAL",
                "Mejoras del Hogar": "HOMEIMPROVEMENT",
                "Consumo Personal": "PERSONAL",
            }
            intent_label = st.selectbox("Destino del Financiamiento", list(intent_map.keys()), index=def_intent_idx)
            loan_intent = intent_map[intent_label]

            st.markdown('<div class="dossier-hdr">III. Historial en Buró de Crédito</div>', unsafe_allow_html=True)
            f7, f8, f9 = st.columns([1, 1.3, 1.1])
            with f7:
                loan_grade = st.selectbox("Grado Buró", ["A", "B", "C", "D", "E", "F", "G"], index=def_grade_idx)
            with f8:
                cb_label = st.selectbox("Impago Previo", ["Limpio (No)", "Con Atraso (Sí)"], index=def_cb_idx)
                cb_person_default_on_file = "Y" if "Sí" in cb_label else "N"
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

    # ----- COLUMNA DERECHA: BENTO GRID DE DICTAMEN, TACÓMETRO Y CASCADA -----
    with right_cockpit:
        # Top Verdict Stamp Banner
        if res["verdict_level"] == "low":
            s_cls, s_bg, s_title = "stamp-approved", "#059669", "CRÉDITO APROBADO"
            s_desc = f"El préstamo genera un retorno neto esperado de +${res['expected_profit']:,.0f} USD con riesgo controlado ({res['pd_selected']*100:.1f}% PD)."
        elif res["verdict_level"] == "mod":
            s_cls, s_bg, s_title = "stamp-cond", "#D97706", "APROBADO CON CONDICIONES (AJUSTAR TASA / MONTO)"
            s_desc = f"Riesgo intermedio ({res['pd_selected']*100:.1f}% PD). Se sugiere ajustar la tasa al menos a {res['break_even_rate']:.1f}% o reducir el monto."
        else:
            s_cls, s_bg, s_title = "stamp-rejected", "#DC2626", "SOLICITUD RECHAZADA POR RIESGO"
            s_desc = f"La pérdida esperada (-${res['expected_loss']:,.0f} USD) supera el ingreso por intereses (+${res['interest_revenue']:,.0f} USD)."

        st.markdown(f"""
        <div class="stamp-banner {s_cls}">
            <div>
                <div style="font-size: 0.73rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #4B5563;">RESOLUCIÓN AUTOMÁTICA DEL COMITÉ · {res['tier']}</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #111827; margin-top: 2px;">{s_title}</div>
                <div style="font-size: 0.84rem; color: #374151; margin-top: 2px;">{s_desc}</div>
            </div>
            <div class="stamp-badge" style="background-color: {s_bg};">SCORE: {res['score_selected']} PTS</div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1 of Bento: Left = Circular FICO Gauge (300-850), Right = 2x2 Financial Ledger
        bento_r1_c1, bento_r1_c2 = st.columns([1.05, 1.15], gap="small")

        with bento_r1_c1:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #064E3B; margin-bottom: 2px;'>Tacómetro de Score Crediticio (Escala 300–850)</div>", unsafe_allow_html=True)
                gauge_color = "#059669" if res["score_selected"] >= 630 else ("#D97706" if res["score_selected"] >= 555 else "#DC2626")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=res["score_selected"],
                    number={"font": {"family": "JetBrains Mono, monospace", "size": 42, "color": "#111827"}, "suffix": " pts"},
                    delta={"reference": 600, "increasing": {"color": "#059669"}, "decreasing": {"color": "#DC2626"}},
                    gauge={
                        "axis": {"range": [300, 850], "tickwidth": 1, "tickcolor": "#6B7280", "tickvals": [300, 450, 555, 630, 750, 850]},
                        "bar": {"color": gauge_color, "thickness": 0.32},
                        "bgcolor": "#F3F4F6",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [300, 555], "color": "#FEE2E2"},
                            {"range": [555, 630], "color": "#FEF3C7"},
                            {"range": [630, 850], "color": "#D1FAE5"},
                        ],
                        "threshold": {
                            "line": {"color": "#064E3B", "width": 3},
                            "thickness": 0.8,
                            "value": 600,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    font={"family": "Plus Jakarta Sans, sans-serif"},
                    paper_bgcolor="#FFFFFF",
                    margin=dict(l=18, r=18, t=22, b=8),
                    height=215,
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Comparativa entre motores · Random Forest: {res['score_rf']} pts | Scorecard WoE: {res['score_lr']} pts (Línea base: 600 pts).")

        with bento_r1_c2:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #064E3B; margin-bottom: 6px;'>Hoja de Balance y Riesgo del Préstamo</div>", unsafe_allow_html=True)
                pnl_color = "#059669" if res["expected_profit"] >= 0 else "#DC2626"
                pnl_sign = "+" if res["expected_profit"] >= 0 else "-"

                st.markdown(f"""
                <div class="ledger-grid">
                    <div class="ledger-box">
                        <div class="ledger-lbl">Prob. de Impago (PD)</div>
                        <div class="ledger-val">{res['pd_selected']*100:.1f}%</div>
                        <div class="ledger-note">RF: {res['pd_rf']*100:.1f}% · WoE: {res['pd_lr']*100:.1f}%</div>
                    </div>
                    <div class="ledger-box">
                        <div class="ledger-lbl">Ratio Deuda / Ingreso</div>
                        <div class="ledger-val">{res['loan_percent_income']*100:.1f}%</div>
                        <div class="ledger-note">${loan_amnt:,.0f} / ${person_income:,.0f}</div>
                    </div>
                    <div class="ledger-box">
                        <div class="ledger-lbl">Interés vs. Pérdida (EL)</div>
                        <div class="ledger-val">+${res['interest_revenue']:,.0f}</div>
                        <div class="ledger-note" style="color: #DC2626;">Pérdida Esp.: -${res['expected_loss']:,.0f}</div>
                    </div>
                    <div class="ledger-box">
                        <div class="ledger-lbl">Utilidad Neta Esperada</div>
                        <div class="ledger-val" style="color: {pnl_color};">{pnl_sign}${abs(res['expected_profit']):,.0f}</div>
                        <div class="ledger-note">Tasa equilibrio: {res['break_even_rate']:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                st.caption("Fórmula bancaria: Pérdida Esperada (EL) = PD × LGD (75%) × Monto Expuesto (EAD).")

        # Row 2 of Bento: Full-Width Financial Waterfall Chart (Score Construction Step by Step)
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #064E3B; margin-bottom: 2px;'>Gráfica de Cascada (Waterfall): ¿Cómo se construyó el Score del cliente paso a paso?</div>", unsafe_allow_html=True)
            st.caption("Muestra cómo cada característica del solicitante suma escalones verdes (+) o resta escalones rojos (-) a partir del puntaje base hasta llegar a su Score.")

            df_imp = res["impacts_df"].copy()
            # Sort by magnitude to display top 6 most impactful features + group rest
            df_imp["abs_pts"] = df_imp["Impacto_Puntos"].abs()
            df_top = df_imp.sort_values(by="abs_pts", ascending=False).head(6)

            # Calculate base intercept score so waterfall lands cleanly on score_lr
            sum_impacts = float(df_top["Impacto_Puntos"].sum())
            base_anchor = round(res["score_lr"] - sum_impacts, 1)

            wf_x = ["Base Inicial"] + df_top["Variable"].tolist() + ["Score Scorecard Final"]
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
                textfont={"family": "JetBrains Mono, monospace", "size": 11, "color": "#111827"},
                connector={"line": {"color": "#9CA3AF", "width": 1, "dash": "dot"}},
                increasing={"marker": {"color": "#059669"}},
                decreasing={"marker": {"color": "#DC2626"}},
                totals={"marker": {"color": "#064E3B"}},
            ))
            y_min_wf = min(base_anchor, res["score_lr"], 300) - 70
            y_max_wf = max(base_anchor, res["score_lr"], 750) + 85
            fig_wf.update_layout(
                font={"family": "Plus Jakarta Sans, sans-serif", "color": "#111827"},
                yaxis_title="Puntos de Score Crediticio",
                yaxis=dict(range=[y_min_wf, y_max_wf], showgrid=True, gridcolor="#F3F4F6"),
                xaxis=dict(showgrid=False, tickangle=-12),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=18, b=10),
                height=290,
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar": False})

# =========================================================================
# VISTA 2: COMITÉ DE CARTERA (SIMULADOR DE RENTABILIDAD DEL BANCO)
# =========================================================================
elif view_mode.startswith("2."):
    st.markdown('<div class="section-title">Comité de Riesgos: Simulador de Política de Aprobación y Utilidad de Cartera</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Simulación financiera sobre los <b>6,482 créditos reales del conjunto de prueba</b>. '
        'Ajusta la política de corte del banco para encontrar el equilibrio exacto entre volumen de colocación y morosidad.</div>',
        unsafe_allow_html=True,
    )

    pol_c1, pol_c2 = st.columns([1, 2.3], gap="medium")

    with pol_c1:
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #064E3B; margin-bottom: 8px;'>Regla de Aprobación de Cartera</div>", unsafe_allow_html=True)
            cutoff_pct = st.slider(
                "Tolerancia Máxima de Impago (Corte PD %)",
                min_value=5,
                max_value=100,
                value=22,
                step=1,
                format="%d%%",
            )
            lgd_pct = st.slider(
                "Severidad de Pérdida en Impago (LGD %)",
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
            <div style="background-color: #ECFDF5; border-left: 4px solid #059669; padding: 12px 14px; border-radius: 6px; font-size: 0.84rem; color: #064E3B; margin-top: 10px; line-height: 1.55;">
                <b>Punto Óptimo Matemático:</b><br>
                La utilidad máxima del banco se logra fijando el corte en <b>PD ≤ {sim['optimal_cutoff_pct']:.0f}%</b>, generando una ganancia neta de <b>+${sim['optimal_max_profit']:,.0f} USD</b>.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #F9FAFB; border: 1px solid #E5E7EB; padding: 12px 14px; border-radius: 6px; font-size: 0.82rem; color: #374151; margin-top: 10px; line-height: 1.6;" class="mono-num">
                <b>Estado de Resultados (Corte {cutoff_pct}%):</b><br>
                • Capital Colocado: <b>${sim['capital_deployed']:,.0f}</b><br>
                • Intereses Cobrados: <b style="color:#059669;">+${sim['interest_earned']:,.0f}</b><br>
                • Capital Perdido (Morosos): <b style="color:#DC2626;">-${sim['principal_lost']:,.0f}</b><br>
                • Utilidad Neta Final: <b>${sim['net_profit']:,.0f}</b>
            </div>
            """, unsafe_allow_html=True)

    with pol_c2:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Créditos Aprobados", f"{sim['approval_rate']:.1f}%", f"{sim['n_approved']:,d} de {sim['total_apps']:,d}")
        with m2:
            st.metric("Morosidad de Cartera", f"{sim['port_default_rate']:.2f}%", f"{sim['port_default_rate'] - sim['base_default_rate']:+.2f}% vs sin modelo", delta_color="inverse")
        with m3:
            st.metric("Utilidad Neta Cartera", f"${sim['net_profit']:,.0f}", f"ROA: {sim['roi_pct']:+.2f}%")
        with m4:
            st.metric("Valor Agregado Modelo", f"+${sim['value_added_vs_base']:,.0f}", f"Sin filtro: ${sim['base_net_profit']:,.0f}")

        with st.container(border=True):
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #064E3B; margin-bottom: 4px;'>Curva de Rentabilidad Neta de la Cartera ($ USD) según Política de Corte</div>", unsafe_allow_html=True)
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=sim["curve"]["cutoff_pct"],
                y=sim["curve"]["net_profit"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.14)",
                line=dict(color="#064E3B", width=3),
                name="Utilidad Neta ($ USD)",
                hovertemplate="<b>Corte PD ≤ %{x}%</b><br>Utilidad Neta: $%{y:,.0f} USD<extra></extra>",
            ))
            fig_area.add_vline(
                x=cutoff_pct,
                line_width=2.5,
                line_dash="dash",
                line_color="#D97706",
                annotation_text=f"Corte seleccionado: {cutoff_pct}%",
                annotation_position="top left",
            )
            fig_area.add_hline(y=0, line_width=1, line_color="#9CA3AF")
            fig_area.update_layout(
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#111827"),
                xaxis_title="Umbral Máximo de Probabilidad de Impago Permitido (%)",
                yaxis_title="Utilidad Neta de la Cartera ($ USD)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=20, b=10),
                height=340,
            )
            st.plotly_chart(fig_area, use_container_width=True, config={"displayModeBar": False})

# =========================================================================
# VISTA 3: AUDITORÍA ESTADÍSTICA (CURVA KS & RANKING IV)
# =========================================================================
else:
    st.markdown('<div class="section-title">Auditoría Estadística: Duelo de Modelos, Curva KS y Poder Predictivo (IV)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Evaluación rigurosa fuera de muestra sobre <b>6,482 préstamos de prueba</b> comparando el estándar regulatorio bancario contra el ensamble no lineal de árboles.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <table class="fin-table">
        <thead>
            <tr>
                <th>Arquitectura Evaluada</th>
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
            <tr class="win-row">
                <td>2. Random Forest Classifier (300 Árboles · Bagging) <span class="win-pill">Ganador</span></td>
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

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    aud_c1, aud_c2 = st.columns([1.1, 1], gap="medium")

    with aud_c1:
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #064E3B; margin-bottom: 4px;'>Curva de Separación Kolmogorov-Smirnov (KS = 72.98%)</div>", unsafe_allow_html=True)
            st.caption("Ordenando a los clientes de mayor a menor riesgo: la curva roja muestra qué tan rápido atrapamos a los morosos y la curva verde a los buenos pagadores. El KS es la brecha máxima entre ambas.")

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
                mode="lines", name="% Morosos Atrapados", line=dict(color="#DC2626", width=3)
            ))
            fig_ks.add_trace(go.Scatter(
                x=df_plot["pct_population"], y=df_plot["cum_goods"],
                mode="lines", name="% Buenos Pagadores", line=dict(color="#059669", width=3)
            ))
            fig_ks.add_vline(
                x=ks_pop_x, line_width=2, line_dash="dash", line_color="#064E3B",
                annotation_text=f"Brecha Máxima KS = {ks_val_max:.1f}%",
                annotation_position="top right"
            )
            fig_ks.update_layout(
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#111827"),
                xaxis_title="% de Clientes Ordenados de Mayor a Menor Riesgo",
                yaxis_title="% Acumulado Capturado",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=30, b=10), height=320
            )
            st.plotly_chart(fig_ks, use_container_width=True, config={"displayModeBar": False})

    with aud_c2:
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #064E3B; margin-bottom: 4px;'>Poder Predictivo por Variable (Information Value - IV)</div>", unsafe_allow_html=True)
            st.caption("Variables con IV > 0.30 (en verde bosque) son los predictores maestros de solvencia crediticia.")

            df_iv_p = engine.iv_table.sort_values(by="Information_Value_IV", ascending=True)
            iv_colors = ["#064E3B" if v >= 0.30 else ("#059669" if v >= 0.10 else "#9CA3AF") for v in df_iv_p["Information_Value_IV"]]

            fig_iv = go.Figure(go.Bar(
                x=df_iv_p["Information_Value_IV"],
                y=df_iv_p["Nombre_Negocio"],
                orientation="h",
                marker_color=iv_colors,
                text=[f"{v:.3f}" for v in df_iv_p["Information_Value_IV"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono, monospace", size=11, color="#111827"),
            ))
            fig_iv.add_vline(x=0.30, line_width=1.5, line_dash="dot", line_color="#D97706", annotation_text="Umbral Fuerte (0.30)")
            fig_iv.update_layout(
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#111827"),
                xaxis_title="Information Value (IV)",
                xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#F3F4F6"),
                yaxis=dict(showgrid=False),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=20, t=10, b=10), height=320
            )
            st.plotly_chart(fig_iv, use_container_width=True, config={"displayModeBar": False})
