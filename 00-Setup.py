# Databricks notebook source
# MAGIC %md
# MAGIC #### Data Setup
# MAGIC - Create an s3 folder
# MAGIC - Copy the `diabetes.csv` file
# MAGIC - Copy the `Postural_Tremor_DA_Raw.csv` file
# MAGIC - Create an iam role that has access to s3 path
# MAGIC - If UC is enabled, Create an external location and storage credential
# MAGIC - If UC is not enabled, Create a cluster with the instance profile with the above IAM role
# MAGIC - Test access

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delta Tables

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Database

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Data Tables

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {demographic_table}")
spark.sql(f"DROP TABLE IF EXISTS {lab_results_table}")
spark.sql(f"DROP TABLE IF EXISTS {physicals_results_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Feature Tables

# COMMAND ----------

#Drop the delta table
spark.sql(f"DROP TABLE IF EXISTS {feature_table_name}")

#Remove the feature store entry
from databricks import feature_store
fs = feature_store.FeatureStoreClient()
try:
  #Check if feature table exists. Delete if exists
  fs.drop_table(name=feature_table_name)
except:
  print(f"Feature table {feature_table_name} not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Inference Tables

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_nonfs}")
spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_fs}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Copy Images for Markdown

# COMMAND ----------

# MAGIC %cd

# COMMAND ----------

src_folder = f"file:///Workspace/Users/{user_email}/merck_diabetes_prediction/_resources"
tgt_folder = "/FileStore/tmp/merck_diabetes_prediction"

dbutils.fs.mkdirs(tgt_folder)
dbutils.fs.cp(src_folder,tgt_folder, True)

# COMMAND ----------

dbutils.fs.ls(tgt_folder)

# COMMAND ----------



# COMMAND ----------


