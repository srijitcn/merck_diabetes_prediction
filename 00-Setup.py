# Databricks notebook source
# MAGIC %md
# MAGIC #### Data
# MAGIC - Create an s3 folder
# MAGIC - copy the diabetes.csv
# MAGIC - Create an iam role that has access to s3 path
# MAGIC - Create an external location and storage credential
# MAGIC - Test access

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delta Tables

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {demographic_table}")
spark.sql(f"DROP TABLE IF EXISTS {lab_results_table}")
spark.sql(f"DROP TABLE IF EXISTS {physicals_results_table}")

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {feature_table_name}")

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_nonfs}")
spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_fs}")
