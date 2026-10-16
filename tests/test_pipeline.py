"""Teste local do treino, sem Databricks.

Reproduz em pandas a limpeza da camada silver (02_silver.py) e roda o treino
completo com um MLflow local.
"""

import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from churn.features import FEATURE_COLS  # noqa: E402
from churn.modelos import treinar  # noqa: E402


def silver_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates("customerID")
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].str.strip()
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = np.where(df["tenure"] == 0, 0.0, total)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


@pytest.fixture(scope="module")
def silver():
    return silver_pandas(pd.read_csv(RAIZ / "data" / "telco_customer_churn.csv"))


def test_silver_sem_nulos(silver):
    assert len(silver) == 7043
    assert silver[FEATURE_COLS].isna().sum().sum() == 0
    assert set(silver["Churn"].unique()) == {0, 1}


def test_treino_completo(silver, tmp_path):
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("churn-teste")
    resultados = treinar(silver, cv=3)

    assert [r.nome for r in resultados]
    melhor = resultados[0]
    print({r.nome: (round(r.cv_f2, 3), {k: round(v, 3) for k, v in r.metricas_validacao.items()}) for r in resultados})
    assert melhor.metricas_validacao["recall"] > 0.7
    assert melhor.metricas_validacao["roc_auc"] > 0.8

    # O modelo salvo no MLflow precisa aceitar os dados crus da silver.
    carregado = mlflow.sklearn.load_model(f"runs:/{melhor.run_id}/model")
    proba = carregado.predict_proba(silver[FEATURE_COLS].head(10))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
