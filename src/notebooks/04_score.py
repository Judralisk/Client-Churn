# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Score (gold)
# MAGIC Aplica o modelo `champion` nos clientes ativos e grava a probabilidade de churn
# MAGIC e a faixa de risco na tabela `gold_previsoes_churn` (armazenada no S3).

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
import mlflow.sklearn
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))
from churn.features import ID_COL, separar_x_y

mlflow.set_registry_uri("databricks-uc")
nome_modelo = f"{catalog}.{schema}.churn_model"
versao = mlflow.MlflowClient().get_model_version_by_alias(nome_modelo, "champion").version
modelo = mlflow.sklearn.load_model(f"models:/{nome_modelo}@champion")

# COMMAND ----------

# Só faz sentido prever churn de quem ainda é cliente.
ativos = spark.table(f"`{catalog}`.`{schema}`.silver_clientes").filter("Churn = 0").toPandas()

x, _ = separar_x_y(ativos)

previsoes = pd.DataFrame(
    {
        ID_COL: ativos[ID_COL],
        "prob_churn": modelo.predict_proba(x)[:, 1],
        "previsao_churn": modelo.predict(x).astype(int),
    }
)
previsoes["faixa_risco"] = pd.cut(
    previsoes["prob_churn"], bins=[0, 0.4, 0.7, 1.0], labels=["baixo", "medio", "alto"], include_lowest=True
).astype(str)
previsoes["versao_modelo"] = str(versao)

# COMMAND ----------

from pyspark.sql import functions as F

(
    spark.createDataFrame(previsoes)
    .withColumn("calculado_em", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"`{catalog}`.`{schema}`.gold_previsoes_churn")
)

display(spark.table(f"`{catalog}`.`{schema}`.gold_previsoes_churn").groupBy("faixa_risco").count())
