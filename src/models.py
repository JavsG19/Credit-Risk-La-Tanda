"""
Credit Risk Modeling Engine:
1. Weight of Evidence (WoE) & Information Value (IV) Feature Engineering (Zero Leakage)
2. Model 1: Regulated Banking Scorecard (WoE Logistic Regression)
3. Model 2: Non-Linear Tree Ensemble (Random Forest Classifier)
4. Banking Evaluation Benchmark: ROC-AUC, Gini, Kolmogorov-Smirnov (KS), Brier Score
5. FICO-Style Credit Score Scaling (300 to 850 points via Points-to-Double-the-Odds)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    f1_score,
    roc_curve
)

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

FEATURE_LABELS_ES = {
    "loan_percent_income": "Ratio Deuda / Ingreso (Loan-to-Income)",
    "loan_grade": "Calificación Preliminar de Buró (Grado A-G)",
    "loan_int_rate": "Tasa de Interés Asignada (%)",
    "person_income": "Ingreso Anual del Solicitante ($)",
    "person_home_ownership": "Situación de Vivienda (Propia/Renta/Hipoteca)",
    "cb_person_default_on_file": "Impago Histórico en Buró (Sí/No)",
    "loan_intent": "Propósito / Destino del Crédito",
    "loan_amnt": "Monto Solicitado ($)",
    "person_emp_length": "Antigüedad Laboral / Negocio (Años)",
    "person_age": "Edad del Solicitante",
    "cb_person_cred_hist_length": "Antigüedad en Buró de Crédito (Años)",
}


class WoEEncoder:
    """
    Fits quantile bins (for numeric features) and category mappings (for categorical features)
    strictly on training data to compute Weight of Evidence (WoE) and Information Value (IV).
    """

    def __init__(self, n_bins: int = 6):
        self.n_bins = n_bins
        self.bin_edges_ = {}
        self.woe_maps_ = {}
        self.iv_dict_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df_tmp = X.copy()
        target = y.values
        total_bad = np.sum(target == 1)
        total_good = np.sum(target == 0)

        for col in NUMERIC_COLS:
            # Learn internal quantile cut points on train set
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(df_tmp[col].dropna(), quantiles))
            full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
            self.bin_edges_[col] = full_edges

            binned = pd.cut(df_tmp[col], bins=full_edges, include_lowest=True)
            self._compute_woe_iv(col, binned, target, total_good, total_bad)

        for col in CATEGORICAL_COLS:
            binned = df_tmp[col].astype(str)
            self._compute_woe_iv(col, binned, target, total_good, total_bad)

        return self

    def _compute_woe_iv(self, col, binned_series, target, total_good, total_bad):
        df_b = pd.DataFrame({"bin": binned_series, "y": target})
        grouped = df_b.groupby("bin", observed=False)["y"].agg(
            total="count", bad="sum"
        )
        grouped["good"] = grouped["total"] - grouped["bad"]

        # Laplace smoothing (+0.5) to prevent log(0)
        grouped["dist_good"] = (grouped["good"] + 0.5) / (total_good + 0.5 * len(grouped))
        grouped["dist_bad"] = (grouped["bad"] + 0.5) / (total_bad + 0.5 * len(grouped))
        grouped["woe"] = np.log(grouped["dist_bad"] / grouped["dist_good"])
        grouped["iv_comp"] = (grouped["dist_bad"] - grouped["dist_good"]) * grouped["woe"]

        self.woe_maps_[col] = grouped["woe"].to_dict()
        self.iv_dict_[col] = float(grouped["iv_comp"].sum())

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_woe = pd.DataFrame(index=X.index)
        for col in NUMERIC_COLS:
            binned = pd.cut(X[col], bins=self.bin_edges_[col], include_lowest=True)
            X_woe[col] = binned.map(self.woe_maps_[col]).astype(float).fillna(0.0)

        for col in CATEGORICAL_COLS:
            binned = X[col].astype(str)
            X_woe[col] = binned.map(self.woe_maps_[col]).astype(float).fillna(0.0)

        return X_woe


def compute_ks_statistic(y_true, y_prob) -> float:
    """
    Computes the Kolmogorov-Smirnov (KS) statistic:
    Maximum separation between Cumulative Distribution of Goods and Bads.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def pd_to_credit_score(prob_default, base_score=600, base_odds=19.0, pdo=50.0):
    """
    Translates Probability of Default (PD) into a standard 300-850 Credit Score
    using Points-to-Double-the-Odds (PDO = 50, Base 600 at 5% PD / 19:1 Good:Bad odds).
    """
    p = np.clip(np.asarray(prob_default, dtype=float), 1e-5, 1.0 - 1e-5)
    good_bad_odds = (1.0 - p) / p
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    raw_score = offset + factor * np.log(good_bad_odds)
    return np.clip(np.round(raw_score), 300, 850).astype(int)


def encode_rf_features(df: pd.DataFrame, fit_columns: list = None):
    """
    Prepares tabular features for Random Forest (numeric + ordinal grade + one-hot categoricals).
    """
    out = df[NUMERIC_COLS].copy()
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    out["loan_grade_num"] = df["loan_grade"].map(grade_map).fillna(3).astype(int)
    out["cb_default_num"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    dummies = pd.get_dummies(
        df[["person_home_ownership", "loan_intent"]],
        prefix=["home", "intent"],
        dtype=int
    )
    out = pd.concat([out, dummies], axis=1)

    if fit_columns is not None:
        for c in fit_columns:
            if c not in out.columns:
                out[c] = 0
        out = out[fit_columns]

    return out


def train_and_benchmark(clean_csv_path: str, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(clean_csv_path)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["loan_status"].astype(int).copy()

    # Stratified 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("=" * 72)
    print(f"Split Estratificado: Train = {len(X_train):,d} créditos | Test = {len(X_test):,d} créditos")
    print(f"Tasa Impago Train: {y_train.mean()*100:.2f}% | Tasa Impago Test: {y_test.mean()*100:.2f}%")
    print("=" * 72)

    # 1. Fit WoE & IV strictly on Train Set
    woe_encoder = WoEEncoder(n_bins=6)
    woe_encoder.fit(X_train, y_train)

    iv_rows = []
    for col, iv_val in sorted(woe_encoder.iv_dict_.items(), key=lambda x: x[1], reverse=True):
        if iv_val >= 0.30:
            strength = "Muy Fuerte (Predictor Clave)"
        elif iv_val >= 0.10:
            strength = "Fuerte"
        elif iv_val >= 0.02:
            strength = "Moderado"
        else:
            strength = "Débil"
        iv_rows.append({
            "Variable": col,
            "Nombre_Negocio": FEATURE_LABELS_ES.get(col, col),
            "Information_Value_IV": round(iv_val, 4),
            "Poder_Predictivo": strength
        })
    df_iv = pd.DataFrame(iv_rows)
    df_iv.to_csv(os.path.join(models_dir, "iv_summary.csv"), index=False)

    print("\n--- RANKING DE VARIABLES POR INFORMATION VALUE (IV - ESTÁNDAR BANCARIO) ---")
    print(df_iv.to_string(index=False))

    # Transform Train & Test with WoE
    X_train_woe = woe_encoder.transform(X_train)
    X_test_woe = woe_encoder.transform(X_test)

    # 2. Model 1: Logistic Regression WoE Scorecard
    lr_woe = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr_woe.fit(X_train_woe, y_train)
    prob_lr = lr_woe.predict_proba(X_test_woe)[:, 1]

    # 3. Model 2: Random Forest Classifier (300 trees)
    X_train_rf = encode_rf_features(X_train)
    rf_cols = list(X_train_rf.columns)
    X_test_rf = encode_rf_features(X_test, fit_columns=rf_cols)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_rf, y_train)
    prob_rf = rf_model.predict_proba(X_test_rf)[:, 1]

    # Evaluate both models
    benchmark_rows = []
    for m_name, probs in [
        ("Regresión Logística (WoE Scorecard)", prob_lr),
        ("Random Forest (300 Árboles)", prob_rf),
    ]:
        auc = roc_auc_score(y_test, probs)
        gini = 2.0 * auc - 1.0
        ks = compute_ks_statistic(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        preds_bin = (probs >= 0.35).astype(int)
        acc = accuracy_score(y_test, preds_bin)
        f1 = f1_score(y_test, preds_bin)

        benchmark_rows.append({
            "Modelo": m_name,
            "ROC-AUC": round(auc, 4),
            "Índice Gini": round(gini, 4),
            "Estadístico KS": f"{ks*100:.2f}%",
            "Brier Score": round(brier, 4),
            "Exactitud (Acc)": f"{acc*100:.1f}%",
            "F1-Score": round(f1, 4),
        })

    df_bench = pd.DataFrame(benchmark_rows)
    df_bench.to_csv(os.path.join(models_dir, "benchmark_results.csv"), index=False)

    print("\n" + "=" * 72)
    print("       RESULTADOS DEL BENCHMARK EN TEST SET (6,482 CRÉDITOS)       ")
    print("=" * 72)
    print(df_bench.to_string(index=False))
    print("=" * 72)

    # Store test set curves and portfolio simulation data for the app
    test_sim_df = X_test.copy()
    test_sim_df["loan_status"] = y_test.values
    test_sim_df["pd_logistic_woe"] = np.round(prob_lr, 4)
    test_sim_df["pd_random_forest"] = np.round(prob_rf, 4)
    test_sim_df["score_logistic"] = pd_to_credit_score(prob_lr)
    test_sim_df["score_rf"] = pd_to_credit_score(prob_rf)
    test_sim_df.to_csv(os.path.join(models_dir, "test_portfolio_predictions.csv"), index=False)

    # Save full production artifacts
    artifacts = {
        "woe_encoder": woe_encoder,
        "lr_woe_model": lr_woe,
        "rf_model": rf_model,
        "rf_columns": rf_cols,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "iv_table": df_iv,
        "benchmark_table": df_bench,
    }
    artifact_path = os.path.join(models_dir, "credit_risk_artifacts.joblib")
    joblib.dump(artifacts, artifact_path)
    print(f"\nArtefactos de producción guardados en: {artifact_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_p = os.path.join(base_dir, "data", "processed", "credit_risk_clean.csv")
    m_dir = os.path.join(base_dir, "models")
    train_and_benchmark(clean_p, m_dir)
