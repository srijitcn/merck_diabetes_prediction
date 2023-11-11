# Databricks notebook source
# MAGIC %md
# MAGIC #### Prepare Data

# COMMAND ----------

numeric_columns = ["Age", "BloodPressure", "Insulin", "BMI", "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies", "Glucose"]
non_zero_columns = ["BloodPressure", "SkinThickness" , "BMI"]
categorical_columns = []
label_column = "Outcome"
key_column = "Id"

feature_cols = numeric_columns + categorical_columns

# COMMAND ----------

catalog = "main"
database = "merck_ml_ws"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read Raw Data
# MAGIC We start with the table that has the results we want to use

# COMMAND ----------

patient_lab_data = spark.table(f"{catalog}.{database}.patient_lab_results").select(key_column, label_column)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Lookup
# MAGIC Idea of feature lookup

# COMMAND ----------

displayHTML(
   """
   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
   <div class='mermaid'>
    flowchart LR
      D["`
        **DATA_TABLE**
        id,pid,zid,label
      `"]
      
      T["`
        **TRAINING_DF**
        F1,F2,F3,F4,F5,F6,F7,F8,F9, Label
      `"]
    
     D --> FeatureStoreClient --> T

    subgraph FeatureStoreClient
      F1["`
        **FEATURE_LOOKUP 1**
        id -> F1,F2,F3
      `"]

      F2["`
        **FEATURE_LOOKUP 2**
        pid -> F4,F5,F6
      `"]

      F3["`
        **FEATURE_LOOKUP 3**
        zid -> F7,F8,F9
      `"]
    end
    </div>
    """)

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import FeatureLookup


feature_table = f"{catalog}.{database}.diabetes_features"

fs = feature_store.FeatureStoreClient()

feature_lookups = [
    FeatureLookup(
        table_name=feature_table,
        feature_names= feature_cols,
        lookup_key=[key_column]
    ),
]

training_set = fs.create_training_set(
    df=patient_lab_data,
    feature_lookups=feature_lookups,
    label=label_column,
    exclude_columns=[key_column]
)

training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

from sklearn.model_selection import train_test_split

training_df_pd = training_df.toPandas()
y = training_df_pd[label_column]
x = training_df_pd[feature_cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Model

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import make_column_selector as selector
import pandas as pd

import lightgbm
from lightgbm import LGBMClassifier

def get_model(model_params):
  #Preprocessors
  imputers = []
  imputers.append(
    ("impute_mean", SimpleImputer(missing_values=0), non_zero_columns)
  )

  numerical_pipeline = Pipeline(steps=[
      ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
      ("imputers", ColumnTransformer(imputers)),
      ("standardizer", StandardScaler())
  ])

  numerical_transformers = [
    ("numerical", numerical_pipeline, numeric_columns)
  ]

  #since we have only numerical transformation, we can diretcly use `numerical_transformers` 
  preprocessor = ColumnTransformer(numerical_transformers, remainder="passthrough", sparse_threshold=0)

  #Model
  lgbmc_classifier = LGBMClassifier(**model_params)

  #Pipeline
  model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgbmc_classifier),
    ])
  
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Objective Function
# MAGIC

# COMMAND ----------

import mlflow
import os
from datetime import datetime 
from sklearn.metrics import f1_score,roc_auc_score
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import numpy as np

mlflow.set_registry_uri("databricks-uc")

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

#Create an MLFlow experiment
experiment_tag = f"diabetes_prediction_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
experiment_path = f'/Users/{user_name}/mlflow_experiments/{experiment_tag}'
 
# Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
experiment = mlflow.set_experiment(experiment_path)

def loss_fn(params):

  # Initialize MLFlow
  # Remember this function will run on workers
  os.environ['DATABRICKS_HOST'] = db_host   
  os.environ['DATABRICKS_TOKEN'] = db_token   
  
  with mlflow.start_run(experiment_id=experiment.experiment_id) as mlflow_run:
    #enable sklearn autologging
    mlflow.sklearn.autolog(disable=True,
                           log_input_examples=True,
                           silent=True)
    
    #get the model
    model = get_model(params)

    #Now lets train the model
    model.fit(x_train,y_train)

    #evaluate the model
    preds = model.predict(x_test)

    #the metric we want to minimize
    roc_score = roc_auc_score(preds,y_test)
    f1score = f1_score(preds,y_test)
    
    mlflow.log_metric("roc_score",roc_score)
    mlflow.log_metric("f1score",f1score)
    mlflow.sklearn.log_model(model,artifact_path="model")

    return {"loss": -f1score,"status": STATUS_OK}


# COMMAND ----------

algo=tpe.suggest

#Reference: https://hyperopt.github.io/hyperopt/getting-started/search_spaces/
search_space = {
  "colsample_bytree": hp.uniform("colsample_bytree",0,1),
  "lambda_l1": hp.uniform("lambda_l1",0,0.5), 
  "lambda_l2": hp.uniform("lambda_l2",0,0.5), 
  "learning_rate": hp.lognormal("learning_rate",0, 1), 
  "max_bin": hp.choice('max_bin', np.arange(50, 255, dtype=int)), 
  "max_depth": hp.choice('max_depth', np.arange(10, 20, dtype=int)), 
  "min_child_samples": 35,
  "n_estimators": 181,
  "num_leaves": 90,
  "path_smooth": 82.58514740065912,
  "subsample": 0.7664087623951591,
  "random_state": 50122439,
}

# COMMAND ----------

trials = SparkTrials()
fmin(
  fn=loss_fn,
  space=search_space,
  algo=algo,
  max_evals=5,
  trials=trials)


# COMMAND ----------

best_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=['metrics.f1score DESC']).iloc[0]

# COMMAND ----------

model_uri = f"runs:/{best_run.run_id}/model"
selected_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(x_test,y_test)

# COMMAND ----------

input_example = {
    "Id":1,
}

fs.log_model(
    selected_model,
    signature = signature,
    artifact_path="model",
    flavor=mlflow.sklearn,    
    training_set=training_set,
    registered_model_name="main.merck_ml_ws.ws_diabetes_prediction_fs",
    pip_requirements = ["emoji==2.8.0"],
    input_example=input_example,
)

# COMMAND ----------


