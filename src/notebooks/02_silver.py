# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver
# MAGIC Tipagem e limpeza, com as mesmas regras da análise original:
# MAGIC - `TotalCharges` vazio só acontece com `tenure = 0` (cliente sem fatura ainda) → vira 0.
# MAGIC - "No internet service" / "No phone service" → "No".
# MAGIC - `Churn` vira 0/1.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "churn")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from pyspark.sql import functions as F

bronze = spark.table(f"`{catalog}`.`{schema}`.bronze_clientes")
colunas_texto = [c for c in bronze.columns if not c.startswith("_")]

df = bronze.select(*[F.trim(F.col(c)).alias(c) for c in colunas_texto])
df = df.replace(["No internet service", "No phone service"], ["No", "No"])

silver = (
    df.withColumn("tenure", F.col("tenure").cast("int"))
    .withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
    .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
    .withColumn(
        "TotalCharges",
        F.when(F.col("tenure") == 0, F.lit(0.0)).otherwise(F.col("TotalCharges").cast("double")),
    )
    .withColumn("Churn", (F.col("Churn") == "Yes").cast("int"))
    .dropDuplicates(["customerID"])
)

# COMMAND ----------

# Checagens de qualidade: falha o job se algo vier errado.
nulos = silver.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in silver.columns]
).first().asDict()
com_nulo = {c: n for c, n in nulos.items() if n}
assert not com_nulo, f"Colunas com nulos na silver: {com_nulo}"
assert silver.filter(~F.col("Churn").isin(0, 1)).count() == 0, "Churn fora de 0/1"

(
    silver.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"`{catalog}`.`{schema}`.silver_clientes")
)

display(silver.groupBy("Churn").count())
