# Databricks notebook source
# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Batch Inference Data

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE {inference_data_table} AS
  SELECT * FROM {feature_table_name} LIMIT 50
""")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {inference_data_table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model

# COMMAND ----------



# COMMAND ----------

model_uri = f"models:/{}/model"
selected_model = mlflow.sklearn.load_model(model_uri)
