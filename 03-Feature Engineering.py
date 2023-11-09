# Databricks notebook source
# MAGIC %md
# MAGIC #### Read Input Data and Extract Feature Columns

# COMMAND ----------

catalog = "main"
database = "merck_ml_ws"

# COMMAND ----------

diabetes_data_table = f"{catalog}.{database}.diabetes"

# COMMAND ----------

feature_columns = ["Age","Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Outcome"]
feature_data = spark.table(diabetes_data_table).select("Id", *feature_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create and Save Data to Feature Table

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

feature_table_name = f"{catalog}.{database}.diabetes_features"
fs.create_table(
    name=feature_table_name,
    primary_keys="Id",
    df=feature_data,
)

# COMMAND ----------

fs.write_table(
    name=feature_table_name,
    df=feature_data,
    mode="merge",
)
