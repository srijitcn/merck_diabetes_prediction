# Databricks notebook source
# MAGIC %md
# MAGIC #### Batch and Streaming Inference on Databricks
# MAGIC On Databricks, you can deploy MLflow models for offline (batch and streaming) inference. Databricks recommends that you use MLflow to deploy machine learning models for batch or streaming inference. For general information about working with MLflow models, see [Log, load, register, and deploy MLflow models](https://docs.databricks.com/en/mlflow/models.html).

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Batch Inference Data
# MAGIC Now we will creatre some mock data to use for inference. 
# MAGIC
# MAGIC Since we are not using feature store, the dataframe is expected to have all the features required by the model

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_nonfs}")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {inference_data_table_nonfs} AS
  SELECT * FROM {feature_table_name} LIMIT 50
""")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {inference_data_table_nonfs}"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model
# MAGIC Let us lookup the model thats tagged as `Production` in model registry to use for our batch inference

# COMMAND ----------

env_or_alias = "champion" if uc_enabled else "production"
model_info = get_latest_model_version(registered_model_name_non_fs, env_or_alias)
model_uri = get_model_uri(model_info,env_or_alias)

# COMMAND ----------

#Lets install model dependencies
import mlflow
req_file = mlflow.pyfunc.get_model_dependencies(model_uri)
%pip install -r $req_file

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a PySpark UDF and use it for batch inference
# MAGIC We can create a PySpark UDF from the model you saved to MLflow. For more information, see [Export a python_function model as an Apache Spark UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf).
# MAGIC
# MAGIC Saving the model as a PySpark UDF allows you to run the model to make predictions on a Spark DataFrame.

# COMMAND ----------

# Create the PySpark UDF
print(f"Creating pyfunc from model uri: {model_uri}")
predict_diabetes_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

feature_columns = ["Age", "BloodPressure", "Insulin", "BMI", "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies", "Glucose"]

prediction_df1 = (spark
                  .table(inference_data_table_nonfs)
                  .withColumn("prediction",predict_diabetes_udf(*feature_columns))
  )
display(prediction_df1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Using Pandas UDF

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd
from pyspark.sql.types import StructType, StructField,LongType,IntegerType, DoubleType

schema = StructType([
    StructField("Age", IntegerType()),
    StructField("BloodPressure", IntegerType()),
    StructField("Insulin", IntegerType()),
    StructField("BMI", DoubleType()),
    StructField("SkinThickness", IntegerType()),
    StructField("DiabetesPedigreeFunction", DoubleType()),
    StructField("Pregnancies", IntegerType()),
    StructField("Glucose", IntegerType()),
    StructField("prediction", IntegerType())
])

@pandas_udf(returnType=schema)
def predict_diabetes_pudf(batches:Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
  model = mlflow.sklearn.load_model(model_uri)
  for batch in batches:
    batch["prediction"] = model.predict(batch)
    yield batch


# COMMAND ----------

prediction_df2 = (spark
                  .table(inference_data_table_nonfs)
                  .withColumn("prediction",predict_diabetes_udf(*feature_columns))
  )
display(prediction_df2)

# COMMAND ----------


