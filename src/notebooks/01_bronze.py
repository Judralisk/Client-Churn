# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze
# MAGIC Lê o CSV bruto do S3 sem alterar nada e grava em Delta, com data e arquivo de origem.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "churn")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from pyspark.sql import functions as F

origem = f"/Volumes/{catalog}/{schema}/raw/"

bronze = (
    spark.read.option("header", True)
    .option("inferSchema", False)  # tudo como string na bronze
    .csv(origem)
    .withColumn("_arquivo_origem", F.col("_metadata.file_path"))
    .withColumn("_ingerido_em", F.current_timestamp())
)

(
    bronze.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"`{catalog}`.`{schema}`.bronze_clientes")
)

print(f"{bronze.count()} linhas gravadas em {catalog}.{schema}.bronze_clientes")
