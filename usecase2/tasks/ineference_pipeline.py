import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from usecase2.common import Task
#Visual Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pickle

import boto3
# Importing necessary libraries for encoding
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Importing necessary library for scaling
from sklearn.preprocessing import StandardScaler

# Importing necessary library for train-test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# # Importing necessary libraries for model development and evaluation
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
# import xgboost as xgb
# import lightgbm as lgb

from databricks import feature_store
from pyspark.sql import SparkSession
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from databricks.feature_store import feature_table, FeatureLookup
# from utils import apply_model

warnings.filterwarnings('ignore')
from pyspark.dbutils import DBUtils



class inference(Task):
    pass

    # def inference_data():



    #     model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
                
    #     # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    #     training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label=target,exclude_columns=lookup_key)
    #     df= training_set.load_df().toPandas()
    #     # batch_df has columns ['customer_id', 'account_creation_date']
    #     predictions = fs.score_batch(
    #         'models:/example_model/1',
    #         batch_df
    #     )
    #     return batch_df
    

    # def Shapl(X_test,):
    #     import shap

    #     # Create a SHAP explanation
    #     explainer = shap.Explainer(conversion_classifer, X_val)
    #     shap_values = explainer(X_test)

    #     # Visualize the SHAP explanation
    #     shap.plots.bar(shap_values[1])
    #     shap.save_html('shapfig.html')