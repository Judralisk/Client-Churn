"""Definição das features e do pré-processamento usado no treino e no score."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo de um DataFrame da camada silver."""
    x = df[FEATURE_COLS].copy()
    x[NUMERIC_COLS] = x[NUMERIC_COLS].astype("float64")
    y = df[TARGET_COL].astype("int64")
    return x, y


def preprocessador() -> ColumnTransformer:
    """Padroniza as numéricas e aplica one-hot nas categóricas.

    O one-hot substitui o mapeamento manual (No=0, Yes=1, DSL=1...) da análise
    original e aceita categorias novas no score sem quebrar.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
