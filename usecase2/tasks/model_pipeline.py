import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from usecase2.common import Task
#Visual Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

import boto3
# Importing necessary libraries for encoding
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Importing necessary library for scaling
from sklearn.preprocessing import StandardScaler

# Importing necessary library for train-test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Importing necessary libraries for model development and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter Tuning
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

from databricks import feature_store
from pyspark.sql import SparkSession
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from databricks.feature_store import feature_table, FeatureLookup
# from utils import apply_model

warnings.filterwarnings('ignore')
from pyspark.dbutils import DBUtils

fs = feature_store.FeatureStoreClient()


class model_training(Task):

    def train_test_val_split(self,target,test_split,val_split,table_name,lookup_key,inference_data_df):


        model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
                
        # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
        training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label=target,exclude_columns=lookup_key)
        df= training_set.load_df().toPandas()

        # X_train, X_val, y_train, y_val,X_test,y_test=self.train_test_val_split(df_input,test_split,val_split)
        

        X = df.drop(target, axis=1)
        y = df[target]

        # Performing the train-test split
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify= y)

        # Performing the train-val split using train data
        X_train, X_val, y_train, y_val = train_test_split(X_train_pre, y_train_pre, test_size=val_split, random_state=42, stratify= y_train_pre)
        return X_train, X_val, y_train, y_val,X_test,y_test
    
    def metrics(self,y_train,y_pred_train,y_val,y_pred_val,y_test,y_pred):

        f1_train = f1_score(y_train, y_pred_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        f1_val = f1_score(y_val, y_pred_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)

        f1_val = f1_score(y_test, y_pred)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        
        recall_train=recall_score(y_train, y_pred_train)
        recall_val=recall_score(y_val, y_pred_val)
    
        return {'accuracy_train': round(accuracy_train, 2),'accuracy_val': round(accuracy_val, 2),
                'f1 score train': round(f1_train, 2), 'f1 score val': round(f1_val, 2)}
   

        def confusion_metrics(true_labels, predicted_labels):
            """
            Logs confusion metrics and classification report in MLflow.

            Parameters:
            - true_labels: The true labels (ground truth).
            - predicted_labels: The predicted labels (model predictions).
            - run_name: The name for the MLflow run.

            Returns:
            - None
            """
            # Calculate the confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)

            # Log the confusion matrix as an artifact in MLflow
            # with mlflow.start_run(run_name=run_name):
            #     mlflow.log_artifact('confusion_matrix.png')

            # Calculate and log additional classification metrics
            classification_metrics = classification_report(true_labels, predicted_labels, output_dict=True)
            # for metric_name, metric_value in classification_metrics.items():
            #     mlflow.log_metric(metric_name, metric_value)
            
            # Log the confusion matrix plot as an artifact in MLflow
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig('confusion_matrix.png')
            # mlflow.log_artifact('confusion_matrix.png')
            return cm,classification_report

        

    
    def train_model(self):
        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        dbutils = DBUtils(spark)

        aws_access_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-secret-key")
        
        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                aws_secret_access_key=aws_secret_key, 
                region_name='ap-south-1')
        
        bucket_name =  self.conf['s3']['bucket_name']
        csv_file_key = self.conf['cleaned_data']['final_features_df_path']
        print('start')

        
        s3_object = s3.Object(bucket_name, csv_file_key)
        
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))

        

        df_input_spark = spark.createDataFrame(df_input)

        inference_data_df = df_input_spark.select(self.conf['features']['id_target_col_list'])



        target=self.conf['features']['target_col']

        X_train, X_val, y_train, y_val,X_test,y_test=self.train_test_val_split(target,self.conf['split']['test_split'],self.conf['split']['val_split'],self.conf['feature-store']['table_name'],self.conf['feature-store']['lookup_key'],inference_data_df)
        mlflow.set_experiment(self.conf['Mlflow']['experiment_name'])
        with mlflow.start_run() as run:


            # best_param = {'colsample_bytree': 0.8011137517906433, 'gamma': 0.0003315092691686855,
            # 'max_depth': 7, 'reg_alpha': 0.20064996416845873, 'subsample': 0.19265865309365698}
            model_xgb = xgb.XGBClassifier(self.conf['param'])
            model_xgb.fit(X_train.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'), y_train)
            y_pred_train = model_xgb.predict(X_train.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_pred_val = model_xgb.predict(X_val.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_pred_test = model_xgb.predict(X_test.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            
            fpr, tpr, threshold = roc_curve(y_test,y_pred_test)
            roc_auc = auc(fpr, tpr)
            cm=self.confusion_matrix(y_test,y_pred_test)
            mlflow.log_artifact('confusion_matrix.png')

            
            
            mlflow.log_metric("roc_auc",roc_auc)
            
            mlflow.log_metrics(self.metrics(y_train,y_pred_train,y_val,y_pred_val,y_test,y_pred_test))

            mlflow.xgboost.log_model(model=model_xgb,artifact_path="usecase2",registered_model_name="Physician Model")
            # fs.log_model(
            #                     model=LR_Classifier,
            #                     artifact_path="health_prediction",
            #                     flavor=mlflow.sklearn,
            #                     training_set=training_set,
            #                     registered_model_name="pharma_model",
            #                     )

    def launch(self):
        self.logger.info("Launching sample ETL task")
        self.train_model()
        self.logger.info("Sample ETL task finished!")

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = model_training()

    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()


            # fs.log_model(
            # model=LR_Classifier,
            # artifact_path="health_prediction",
            # flavor=mlflow.sklearn,
            # training_set=training_set,
            # registered_model_name="pharma_usecase2_model",
            # )
        
           
    # def train_val_test_split(self, table_name, lookup_key,target, inference_data_df):
    #      # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    #         model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
        
    #         # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    #         training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label=target,exclude_columns=lookup_key)
    #         training_pd = training_set.load_df().toPandas()
        
    #         # Create train and test datasets
    #         X = training_pd.drop(target, axis=1)
    #         y = training_pd[target]
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.conf['ModelTraining']['test_split'], random_state=42)

    #         X_train_pre, X_val, y_train_pre, y_val = train_test_split(X_train, y_train, test_size=self.conf['ModelTraining']['validation_split'], random_state=43)
    #         return X_train_pre, X_test, y_train_pre, y_test, X_val, y_val, training_set
    # def apply_model(self,model, X_train, y_train, X_val, y_val, drop_id_col_list):
    # # Fit the model
    #         mlflow.set_experiment(self.conf['Mlflow']['experiment_name'])
    #         with mlflow.start_run(run_name=self.conf['Mlflow']['run_name']) as run:
    #             model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)

    #             # Make predictions
    #             y_train_pred = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
    #             y_pred = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

    #             # Calculate performance metrics
    #             accuracy_train = accuracy_score(y_train, y_train_pred)
    #             accuracy_val = accuracy_score(y_val, y_pred)
    #             f1_train = f1_score(y_train,y_train_pred)
    #             f1_val = f1_score(y_val, y_pred)
    #             return accuracy_train, accuracy_val,f1_train,f1_val
    # def train_vanilla_models():
    #       # Defining the models
    #         vanila_models = [
    #             ("Logistic Regression", LogisticRegression(random_state=321)),
    #             ("Decision Tree", DecisionTreeClassifier(random_state=321)),
    #             ("Random Forest", RandomForestClassifier(random_state=321)),
    #             ("XGB Classifier", xgb.XGBClassifier(random_state=321)),
    #             ("LGBM Classifier", lgb.LGBMClassifier(random_state=321))
    #         ]
    #         X_train_pre, X_test, y_train_pre, y_test, X_val, y_val, training_set=self.train_val_test_split(table_name, lookup_key,target, inference_data_df)
    #         for name, model in vanila_models:
    #                 apply_model(model, X_train, y_train, X_val, y_val,drop_id_col_list)

    # def evaluate_model():
