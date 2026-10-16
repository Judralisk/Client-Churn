"""Treino, comparação e avaliação dos modelos de churn com MLflow."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.features import preprocessador, separar_x_y

SEED = 42

# F2 dá mais peso ao recall que à precisão: perder um cliente que ia sair custa
# mais do que oferecer retenção a quem ficaria. A análise original otimizava só
# recall, o que premia modelos que marcam todo mundo como churn.
F2 = make_scorer(fbeta_score, beta=2)


def candidatos(proporcao_neg_pos: float) -> dict[str, tuple[object, dict]]:
    """Modelos e grades de hiperparâmetros testados.

    O desbalanceamento (~26% de churn) é tratado com pesos de classe, no lugar
    do oversampling da versão original, para não duplicar linhas antes do CV.
    """
    return {
        "regressao_logistica": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"modelo__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {
                "modelo__n_estimators": [300, 600],
                "modelo__max_depth": [6, 10, None],
                "modelo__min_samples_leaf": [1, 5],
            },
        ),
        "xgboost": (
            XGBClassifier(
                scale_pos_weight=proporcao_neg_pos,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            ),
            {
                "modelo__n_estimators": [200, 400],
                "modelo__max_depth": [3, 5],
                "modelo__learning_rate": [0.03, 0.1],
            },
        ),
    }


def metricas(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


@dataclass
class Resultado:
    nome: str
    run_id: str
    cv_f2: float
    metricas_validacao: dict[str, float]
    modelo: Pipeline


def treinar(df_silver: pd.DataFrame, cv: int = 5) -> list[Resultado]:
    """Treina todos os candidatos, registra cada um como run do MLflow e
    retorna os resultados ordenados pelo F2 da validação cruzada."""
    x, y = separar_x_y(df_silver)
    # Hold-out 70/30 estratificado, como na análise original.
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=SEED
    )
    proporcao = float((y_tr == 0).sum() / (y_tr == 1).sum())

    resultados: list[Resultado] = []
    for nome, (estimador, grade) in candidatos(proporcao).items():
        with mlflow.start_run(run_name=nome) as run:
            pipe = Pipeline([("prep", preprocessador()), ("modelo", estimador)])
            busca = GridSearchCV(pipe, grade, scoring=F2, cv=cv, n_jobs=-1)
            busca.fit(x_tr, y_tr)
            melhor = busca.best_estimator_

            y_pred = melhor.predict(x_val)
            y_proba = melhor.predict_proba(x_val)[:, 1]
            m = metricas(y_val, y_pred, y_proba)

            mlflow.log_params({k.replace("modelo__", ""): v for k, v in busca.best_params_.items()})
            mlflow.log_param("modelo", nome)
            mlflow.log_metric("cv_f2", busca.best_score_)
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})

            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_val, y_pred, display_labels=["Não churn", "Churn"], ax=ax, colorbar=False
            )
            ax.set_title(nome)
            mlflow.log_figure(fig, "matriz_confusao.png")
            plt.close(fig)

            assinatura = infer_signature(x_tr.head(50), melhor.predict_proba(x_tr.head(50)))
            mlflow.sklearn.log_model(
                melhor,
                "model",
                signature=assinatura,
                input_example=x_tr.head(5),
                # cloudpickle funciona igual em qualquer versão do MLflow do Databricks
                serialization_format="cloudpickle",
            )

            resultados.append(Resultado(nome, run.info.run_id, busca.best_score_, m, melhor))

    return sorted(resultados, key=lambda r: r.cv_f2, reverse=True)


def registrar_campeao(melhor: Resultado, nome_modelo_uc: str) -> str:
    """Registra o melhor modelo no Unity Catalog e aponta o alias 'champion' para ele."""
    mlflow.set_registry_uri("databricks-uc")
    versao = mlflow.register_model(f"runs:/{melhor.run_id}/model", nome_modelo_uc)
    cliente = mlflow.MlflowClient(registry_uri="databricks-uc")
    cliente.set_registered_model_alias(nome_modelo_uc, "champion", versao.version)
    return versao.version
