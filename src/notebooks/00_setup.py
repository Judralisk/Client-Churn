# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup no Unity Catalog
# MAGIC Cria o schema com armazenamento no **S3** e o volume que aponta para os dados brutos.
# MAGIC Pré-requisito: uma *external location* do Unity Catalog cobrindo o bucket (ver README).

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "churn")
dbutils.widgets.text("bucket", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
bucket = dbutils.widgets.get("bucket")
assert catalog and bucket, "Preencha os parâmetros catalog e bucket"

# COMMAND ----------

# Tabelas gerenciadas do schema ficam em s3://<bucket>/uc
spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}` "
    f"MANAGED LOCATION 's3://{bucket}/uc'"
)

# Volume externo com o CSV bruto em s3://<bucket>/raw
spark.sql(
    f"CREATE EXTERNAL VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.raw "
    f"LOCATION 's3://{bucket}/raw'"
)

display(dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/raw"))
