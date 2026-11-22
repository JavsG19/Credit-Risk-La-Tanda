"""
Production Inference & Portfolio Economics Engine for Credit Risk Scoring
Calculates applicant PD, 300-850 Credit Score, feature-level point contributions,
loan-level Expected Loss (EL = PD * LGD * EAD), and portfolio-wide cutoff optimization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from models import (
    encode_rf_features,
    pd_to_credit_score,
    train_and_benchmark,
    FEATURE_LABELS_ES,
)


class CreditRiskEngine:
    def __init__(self, models_dir: str, clean_csv_path: str):
        self.models_dir = models_dir
        self.clean_csv_path = clean_csv_path
        artifact_path = os.path.join(models_dir, "credit_risk_artifacts.joblib")
        sim_csv_path = os.path.join(models_dir, "test_portfolio_predictions.csv")

        # Self-healing loader for cross-platform cloud deployment (Python 3.12 vs 3.14)
        try:
            if not os.path.exists(artifact_path) or not os.path.exists(sim_csv_path):
                raise FileNotFoundError("Artifacts missing, building...")
            self.artifacts = joblib.load(artifact_path)
            # Test a dummy prediction to ensure binary compatibility
            _ = self.artifacts["rf_model"].n_estimators
        except Exception:
            if not os.path.exists(clean_csv_path):
                from data_prep import download_and_prepare_data
                raw_p = os.path.join(os.path.dirname(os.path.dirname(clean_csv_path)), "raw", "credit_risk_raw.csv")
                download_and_prepare_data(raw_p, clean_csv_path)
            train_and_benchmark(clean_csv_path, models_dir)
            self.artifacts = joblib.load(artifact_path)

        self.woe_encoder = self.artifacts["woe_encoder"]
        self.lr_model = self.artifacts["lr_woe_model"]
        self.rf_model = self.artifacts["rf_model"]
        self.rf_columns = self.artifacts["rf_columns"]
        self.iv_table = self.artifacts["iv_table"]
        self.benchmark_table = self.artifacts["benchmark_table"]
        self.test_portfolio = pd.read_csv(sim_csv_path)

    def evaluate_applicant(
        self,
        person_age: int,
        person_income: float,
        person_home_ownership: str,
        person_emp_length: float,
        loan_intent: str,
        loan_grade: str,
        loan_amnt: float,
        loan_int_rate: float,
        cb_person_default_on_file: str,
        cb_person_cred_hist_length: int,
        lgd: float = 0.75,
        active_model: str = "Random Forest (300 Árboles)",
    ) -> dict:
        loan_percent_income = round(loan_amnt / max(person_income, 1.0), 4)

        row_df = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }])

        # 1. WoE Logistic Scorecard prediction & factor breakdown
        X_woe = self.woe_encoder.transform(row_df)
        pd_lr = float(self.lr_model.predict_proba(X_woe)[0, 1])
        score_lr = int(pd_to_credit_score(pd_lr))

        # 2. Random Forest prediction
        X_rf = encode_rf_features(row_df, fit_columns=self.rf_columns)
        pd_rf = float(self.rf_model.predict_proba(X_rf)[0, 1])
        score_rf = int(pd_to_credit_score(pd_rf))

        if "Random Forest" in active_model:
            pd_selected = pd_rf
            score_selected = score_rf
        else:
            pd_selected = pd_lr
            score_selected = score_lr

        # 3. Feature-level impact points (from WoE * beta scaled by PDO factor)
        pdo_factor = 50.0 / np.log(2.0)
        coefs = self.lr_model.coef_[0]
        woe_vals = X_woe.iloc[0].values
        feature_cols = list(X_woe.columns)

        impacts = []
        for col, beta, w_val in zip(feature_cols, coefs, woe_vals):
            # Negative log-odds of default adds positive points to credit score
            score_pts = float(-beta * w_val * pdo_factor)
            impacts.append({
                "Variable": FEATURE_LABELS_ES.get(col, col),
                "Impacto_Puntos": round(score_pts, 1),
                "Efecto": "Factor Protector (+)" if score_pts >= 0 else "Factor de Riesgo (-)"
            })
        df_impacts = pd.DataFrame(impacts).sort_values(by="Impacto_Puntos", ascending=True)

        # 4. Loan Financial Economics (Expected Return vs. Expected Loss)
        # Expected Interest Revenue if performing = loan_amnt * (loan_int_rate / 100)
        interest_revenue = loan_amnt * (loan_int_rate / 100.0)
        # Expected Loss (EL) = PD * LGD * EAD
        expected_loss = pd_selected * lgd * loan_amnt
        # Expected Profit = (1 - PD) * Interest_Revenue - Expected_Loss
        expected_profit = (1.0 - pd_selected) * interest_revenue - expected_loss
        # Break-even interest rate where Expected Profit == 0:
        break_even_rate = ((pd_selected * lgd) / max(1.0 - pd_selected, 1e-4)) * 100.0

        # Risk Tier & Decision Verdict
        if pd_selected <= 0.08:
            tier = "AAA · Riesgo Mínimo"
        elif pd_selected <= 0.16:
            tier = "AA · Riesgo Bajo"
        elif pd_selected <= 0.25:
            tier = "A · Riesgo Moderado"
        elif pd_selected <= 0.40:
            tier = "B · Riesgo Elevado"
        else:
            tier = "C · Riesgo Crítico"

        if pd_selected <= 0.20 and expected_profit > 0:
            verdict = "APROBADO"
            verdict_level = "low"
        elif pd_selected <= 0.35 and break_even_rate <= 24.0:
            verdict = "APROBADO CON CONDICIONES (AJUSTE DE TASA / MONTO)"
            verdict_level = "mod"
        else:
            verdict = "RECHAZADO"
            verdict_level = "high"

        return {
            "pd_selected": pd_selected,
            "score_selected": score_selected,
            "pd_rf": pd_rf,
            "score_rf": score_rf,
            "pd_lr": pd_lr,
            "score_lr": score_lr,
            "loan_percent_income": loan_percent_income,
            "tier": tier,
            "verdict": verdict,
            "verdict_level": verdict_level,
            "interest_revenue": interest_revenue,
            "expected_loss": expected_loss,
            "expected_profit": expected_profit,
            "break_even_rate": break_even_rate,
            "impacts_df": df_impacts,
        }

    def simulate_portfolio_policy(
        self,
        max_pd_cutoff: float = 0.25,
        lgd: float = 0.75,
        active_model: str = "Random Forest (300 Árboles)",
    ) -> dict:
        df_sim = self.test_portfolio.copy()
        pd_col = "pd_random_forest" if "Random Forest" in active_model else "pd_logistic_woe"

        # Baseline (Approving 100% of applications without model)
        total_apps = len(df_sim)
        base_default_rate = df_sim["loan_status"].mean() * 100.0
        df_sim["interest_rev"] = df_sim["loan_amnt"] * (df_sim["loan_int_rate"] / 100.0)
        df_sim["realized_pnl"] = np.where(
            df_sim["loan_status"] == 0,
            df_sim["interest_rev"],
            -df_sim["loan_amnt"] * lgd,
        )
        base_net_profit = float(df_sim["realized_pnl"].sum())

        # Filtered Portfolio under Cutoff Policy (Approve if PD <= max_pd_cutoff)
        approved = df_sim[df_sim[pd_col] <= max_pd_cutoff].copy()
        n_approved = len(approved)
        approval_rate = (n_approved / total_apps) * 100.0

        if n_approved > 0:
            port_default_rate = approved["loan_status"].mean() * 100.0
            capital_deployed = float(approved["loan_amnt"].sum())
            interest_earned = float(approved.loc[approved["loan_status"] == 0, "interest_rev"].sum())
            principal_lost = float((approved.loc[approved["loan_status"] == 1, "loan_amnt"] * lgd).sum())
            net_profit = interest_earned - principal_lost
            roi_pct = (net_profit / max(capital_deployed, 1.0)) * 100.0
        else:
            port_default_rate = 0.0
            capital_deployed = 0.0
            interest_earned = 0.0
            principal_lost = 0.0
            net_profit = 0.0
            roi_pct = 0.0

        # Compute Profit Curve across every 1% integer cutoff (5% to 100%) to match UI slider
        cutoffs = np.arange(5, 101) / 100.0
        curve_cutoffs = []
        curve_profits = []
        curve_def_rates = []

        for c in cutoffs:
            sub = df_sim[df_sim[pd_col] <= c]
            curve_cutoffs.append(int(round(c * 100)))
            curve_profits.append(round(float(sub["realized_pnl"].sum()), 0))
            curve_def_rates.append(round(float(sub["loan_status"].mean() * 100) if len(sub) > 0 else 0.0, 2))

        best_idx = int(np.argmax(curve_profits))
        optimal_cutoff_pct = curve_cutoffs[best_idx]
        optimal_max_profit = curve_profits[best_idx]

        return {
            "total_apps": total_apps,
            "n_approved": n_approved,
            "approval_rate": approval_rate,
            "base_default_rate": base_default_rate,
            "port_default_rate": port_default_rate,
            "capital_deployed": capital_deployed,
            "interest_earned": interest_earned,
            "principal_lost": principal_lost,
            "net_profit": net_profit,
            "base_net_profit": base_net_profit,
            "value_added_vs_base": net_profit - base_net_profit,
            "roi_pct": roi_pct,
            "optimal_cutoff_pct": optimal_cutoff_pct,
            "optimal_max_profit": optimal_max_profit,
            "curve": {
                "cutoff_pct": curve_cutoffs,
                "net_profit": curve_profits,
                "default_rate": curve_def_rates,
            },
        }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m_dir = os.path.join(base_dir, "models")
    clean_p = os.path.join(base_dir, "data", "processed", "credit_risk_clean.csv")
    engine = CreditRiskEngine(m_dir, clean_p)
    res = engine.evaluate_applicant(
        person_age=28,
        person_income=65000,
        person_home_ownership="MORTGAGE",
        person_emp_length=5.0,
        loan_intent="VENTURE",
        loan_grade="B",
        loan_amnt=12000,
        loan_int_rate=11.5,
        cb_person_default_on_file="N",
        cb_person_cred_hist_length=6,
    )
    print("Applicant Test -> Score:", res["score_selected"], "PD:", f"{res['pd_selected']*100:.1f}%", "Verdict:", res["verdict"])
    sim = engine.simulate_portfolio_policy(max_pd_cutoff=0.22)
    print("Portfolio Sim -> Net Profit: $", f"{sim['net_profit']:,.0f}", "vs Base: $", f"{sim['base_net_profit']:,.0f}")
