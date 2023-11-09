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

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS main.merck_ml_ws

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS main.merck_ml_ws.diabetes

# COMMAND ----------


