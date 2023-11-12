# Databricks notebook source
# MAGIC %md
# MAGIC #### Initialization
# MAGIC
# MAGIC We will initialize few variables and declare some utility functions
# MAGIC
# MAGIC We can always invoke this notebook from other notebooks using the `%run` magic command. Variables and methods defined in this notebook will be available in the calling notebook. 
# MAGIC
# MAGIC To know more read the documentation [here](https://docs.databricks.com/en/notebooks/notebook-workflows.html)

# COMMAND ----------

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]
user_prefix = f"{user_name[0:4]}{str(len(user_name)).rjust(3, '0')}"

# COMMAND ----------

uc_enabled = False
catalog = "main"
database = "merck_ml_ws"

demographic_table = f"{database}.patient_demographics"
lab_results_table = f"{database}.patient_lab_results"
physicals_results_table = f"{database}.patient_pysicals"
feature_table_name = f"{database}.diabetes_features_{user_prefix}"
inference_data_table = f"{database}.patient_data_{user_prefix}"

model_registry_uri = "databricks"
registered_model_name_non_fs = f"{user_prefix}_diabetes_prediction_nonfs"
registered_model_name_fs = f"{user_prefix}_diabetes_prediction_fs"

print(f"***************************************************")
print(f"Unity Catalog is { 'Enabled' if uc_enabled else 'Not Enabled' }")
print(" ")

if uc_enabled :
  demographic_table = f"{catalog}.{demographic_table}"
  lab_results_table = f"{catalog}.{lab_results_table}"
  physicals_results_table = f"{catalog}.{physicals_results_table}"
  feature_table_name =  f"{catalog}.{feature_table_name}"
  inference_data_table = f"{catalog}.{inference_data_table}"
  model_registry_uri = "databricks-uc"
  registered_model_name_non_fs = f"{catalog}.{database}.{registered_model_name_non_fs}"
  registered_model_name_fs = f"{catalog}.{database}.{registered_model_name_fs}"  

print(f"Demographic table (demographic_table): {demographic_table}")
print(f"Lab Result table (lab_results_table): {lab_results_table}")
print(f"Physicals Result table (physicals_results_table): {physicals_results_table}")
print(f"Feature table (feature_table_name): {feature_table_name}")
print(f"Inference Data table (inference_data_table): {inference_data_table}")
print(" ")
print(f"Model Registry URI (model_registry_uri): {model_registry_uri}")
print(f"Model Name Non Feature Store (registered_model_name_non_fs): {registered_model_name_non_fs}")
print(f"Model Name With Feature Store (registered_model_name_fs): {registered_model_name_fs}")
print(f"***************************************************")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient
#Gets the 
def get_latest_model_info(model_name: str, env: str):
  client = MlflowClient()
  model = client.get_latest_versions(model_name, stages=[env])[0]
  return model

# COMMAND ----------

get_latest_model_info("srij011_diabetes_prediction_nonfs","None")

# COMMAND ----------


