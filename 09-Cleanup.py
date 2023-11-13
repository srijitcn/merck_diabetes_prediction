# Databricks notebook source
# MAGIC %md
# MAGIC #### Cleanup Data from S3 bucket
# MAGIC Please cleanup the S3 bucket manually

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delta Tables

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Database

# COMMAND ----------

spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Feature Tables

# COMMAND ----------

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
# MAGIC #### Remove Image copied for Markdown

# COMMAND ----------

#Commenting this so that everyone can refer to these images later
#Uncomment if you want to delete the images

#tgt_folder = "/FileStore/tmp/merck_diabetes_prediction"
#dbutils.fs.rm(tgt_folder, True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delete endpoints

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import *

w = WorkspaceClient(host=db_host,token=db_token)

endpoint_name = f"{registered_model_name_non_fs}_endpoint"
try:
  w.serving_endpoints.delete(name=endpoint_name)  
except:
  print(f"Endpoint {endpoint_name} does not exist")
