"""
Data Acquisition & Cleaning Pipeline for Credit Risk Scoring Engine
Automatically downloads the benchmark Credit Risk Dataset (+32,500 loan records),
audits data quality, removes extreme recording anomalies, and prepares the clean dataset.
"""

import os
import ssl
import urllib.request
import pandas as pd
import numpy as np

DATASET_URLS = [
    "https://raw.githubusercontent.com/LeHongNgoc3820/Project_Credit_Risk/main/credit_risk_dataset.csv",
]

def download_and_prepare_data(raw_csv_path: str, clean_csv_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)

    if not os.path.exists(raw_csv_path):
        ctx = ssl._create_unverified_context()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        downloaded = False
        for url in DATASET_URLS:
            try:
                print(f"Downloading dataset from: {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                    content = resp.read()
                    if b"person_age" in content and len(content) > 50000:
                        with open(raw_csv_path, "wb") as f:
                            f.write(content)
                        downloaded = True
                        print(f"Downloaded successfully ({len(content):,d} bytes).")
                        break
            except Exception as e:
                print(f"Mirror failed ({url}): {e}")
        if not downloaded:
            raise RuntimeError("Could not download dataset from public mirrors.")

    df_raw = pd.read_csv(raw_csv_path)
    raw_count = len(df_raw)
    print("=" * 65)
    print("         REPORTE DE ADQUISICIÓN Y LIMPIEZA DE DATOS (ETL)        ")
    print("=" * 65)
    print(f"Registros originales descargados: {raw_count:,d} solicitudes de crédito")
    print(f"Columnas originales ({len(df_raw.columns)}): {list(df_raw.columns)}")

    # Audit missing values before cleaning
    null_counts = df_raw.isna().sum()
    print("\nValores faltantes detectados en crudo:")
    for col, n_null in null_counts[null_counts > 0].items():
        print(f"  - {col}: {n_null:,d} ({n_null/raw_count*100:.1f}%)")

    # 1. Remove exact duplicates
    df = df_raw.drop_duplicates().copy()
    dups_removed = raw_count - len(df)

    # 2. Filter out impossible data entry errors (e.g. age > 95 or employment length > 60 years)
    outliers_mask = (df["person_age"] > 95) | (df["person_emp_length"] > 60)
    n_outliers = int(outliers_mask.sum())
    df = df[~outliers_mask].copy()

    # 3. Impute missing values cleanly without leakage (using median by loan_grade for interest rate, median for emp_length)
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda s: s.fillna(s.median())
    )
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # 4. Feature standardization & Spanish readable mappings for UI/reporting
    df = df.reset_index(drop=True)
    df.to_csv(clean_csv_path, index=False)

    default_rate = df["loan_status"].mean() * 100
    print("\nResumen tras depuración:")
    print(f"  - Duplicados exactos eliminados: {dups_removed:,d}")
    print(f"  - Anomalías de captura eliminadas (edad > 95 o antigüedad > 60 años): {n_outliers:,d}")
    print(f"  - Registros limpios finales: {len(df):,d}")
    print(f"  - Tasa global de Impago (Default = 1): {default_rate:.2f}% ({df['loan_status'].sum():,d} impagos)")
    print(f"  - Tasa de Cumplimiento (Pagado = 0): {100 - default_rate:.2f}% ({(df['loan_status'] == 0).sum():,d} pagados)")
    print("=" * 65)

    print("\nPrimeros 5 registros de ejemplo:")
    print(df.head(5).to_string())

    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_p = os.path.join(base_dir, "data", "raw", "credit_risk_raw.csv")
    clean_p = os.path.join(base_dir, "data", "processed", "credit_risk_clean.csv")
    download_and_prepare_data(raw_p, clean_p)
