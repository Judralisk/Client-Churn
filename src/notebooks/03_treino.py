# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Treino com MLflow
# MAGIC Compara Regressão Logística, Random Forest e XGBoost (grid search com CV, métrica F2),
# MAGIC registra cada um como run do MLflow e publica o melhor no Unity Catalog com o alias `champion`.

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "churn")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

import os
import sys

import mlflow

sys.path.insert(0, os.path.abspath(".."))  # pasta src/ do repositório
from churn.modelos import registrar_campeao, treinar

usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/churn")

# COMMAND ----------

silver = spark.table(f"`{catalog}`.`{schema}`.silver_clientes").toPandas()
resultados = treinar(silver)

for r in resultados:
    print(f"{r.nome:22s} cv_f2={r.cv_f2:.3f}  " + "  ".join(f"{k}={v:.3f}" for k, v in r.metricas_validacao.items()))

# COMMAND ----------

nome_modelo = f"{catalog}.{schema}.churn_model"
versao = registrar_campeao(resultados[0], nome_modelo)
print(f"Campeão: {resultados[0].nome} → {nome_modelo} versão {versao} (alias champion)")
