# Databricks notebook source
# MAGIC %md
# MAGIC #### Read Input Data and Extract Feature Columns

# COMMAND ----------

# MAGIC %md
# MAGIC Let us take a look the data tables we have and what we are trying to acheieve

# COMMAND ----------

displayHTML(
   """
   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
   <div class='mermaid'>
    graph TD
      A[patient_demographics]
      B[patient_lab_results]
      C[patient_pysicals]
      D{join}
      Z[diabetes_features]
      E[Patient Registration System] 
      F[Daily Feed Clinical Data]
      G[Daily Batch Physician System]
      E -.-> A
      F -.-> B
      G -.-> C
      A -->|Id| D
      B --> |Id| D
      C --> |Id| D 
      D --> Z
    </div>
    """)

# COMMAND ----------

catalog = "main"
database = "merck_ml_ws"

demographic_table = f"{catalog}.{database}.patient_demographics"
lab_results_table = f"{catalog}.{database}.patient_lab_results"
physicals_results_table = f"{catalog}.{database}.patient_pysicals"

# COMMAND ----------

feature_columns = ["Age", "BloodPressure", "Insulin", "BMI", "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies", "Glucose"]

lab_df = spark.table(lab_results_table)

phys_df = spark.table(physicals_results_table)

feature_data = (spark
                .table(demographic_table)
                .join(lab_df, "Id")
                .join(phys_df,"Id") 
                .select("Id",*feature_columns)              
                )

# COMMAND ----------

display(feature_data)

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

# COMMAND ----------


