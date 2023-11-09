# Databricks notebook source
# MAGIC %pip install Faker 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

data_file_name = "s3://databricks-e2demofieldengwest/external_location_srijit_nair/merck/diabetes.csv"

# COMMAND ----------

from faker import Faker
faker = Faker()

# COMMAND ----------

@udf
def create_first_name():
  return faker.first_name()

@udf
def create_last_name():
  return faker.last_name()

@udf
def create_address():
  return faker.address()

@udf
def create_email():
  return faker.email()

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df = (spark
      .read
      .option("header","true")
      .option("inferSchema","true")
      .csv(data_file_name)
      .withColumn("Id", monotonically_increasing_id())
      .withColumn("FirstName",create_first_name())
      .withColumn("LastName",create_last_name())
      .withColumn("Address",create_address())
      .withColumn("Email",create_email())
      .select("Id",
              "FirstName",
              "LastName",
              "Address",
              "Email",
              "Age",
              "Pregnancies",
              "Glucose",
              "BloodPressure",
              "SkinThickness",
              "Insulin",
              "BMI",
              "DiabetesPedigreeFunction",
              "Outcome"
              )
)


# COMMAND ----------

display(df)

# COMMAND ----------

df.write.option("overwrite","true").saveAsTable("main.merck_ml_ws.diabetes")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM main.merck_ml_ws.diabetes

# COMMAND ----------


