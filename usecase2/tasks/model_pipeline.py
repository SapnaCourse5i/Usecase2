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
        training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label=target)
        df= training_set.load_df().toPandas()

        # X_train, X_val, y_train, y_val,X_test,y_test=self.train_test_val_split(df_input,test_split,val_split)
        

        X = df.drop(target, axis=1)
        y = df[target]

        # Performing the train-test split
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify= y)

        # Performing the train-val split using train data
        X_train, X_val, y_train, y_val = train_test_split(X_train_pre, y_train_pre, test_size=val_split, random_state=42, stratify= y_train_pre)
        return df,X_train, X_val, y_train, y_val,X_test,y_test,training_set
    
    def metrics(self,y_train,y_pred_train,y_val,y_pred_val,y_test,y_pred):

        f1_train = f1_score(y_train, y_pred_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        f1_val = f1_score(y_val, y_pred_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)

        f1_test = f1_score(y_test, y_pred)
        accuracy_test = accuracy_score(y_val, y_pred_val)
        
        recall_train=recall_score(y_train, y_pred_train)
        recall_val=recall_score(y_val, y_pred_val)
    
        return {'accuracy_train': round(accuracy_train, 2),'accuracy_val': round(accuracy_val, 2),'accuracy_test': round(accuracy_test, 2),
                'f1 score train': round(f1_train, 2), 'f1 score val': round(f1_val, 2),'f1 score test': round(f1_test, 2)}
   

    def confusion_metrics(self,y_test,y_pred):
            """
            Logs confusion metrics and classification report in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            - y_pred: The predicted labels (model predictions).
            - run_name: The name for the MLflow run.

            Returns:
            - None
            """
            # Calculate the confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            
            classification_metrics = classification_report(y_test, y_pred, output_dict=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig('confusion_matrix.png')
           
            return cm,classification_report
    def roc_curve(self,y_test, y_prop):
            """
            Logs Roc_auc curve in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            - y_prob: The predicted probabilities of labels (model predictions).
            - run_name: The name for the MLflow run.

            Returns:
            - None
            """
            y_prop = y_prop[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prop)
            roc_auc = roc_auc_score(y_test, y_prop)

            # Create and save the ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            roc_curve_plot_path = "roc_curve.png"
            
            plt.savefig(roc_curve_plot_path)

    
      

        

    
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

        df,X_train, X_val, y_train, y_val,X_test,y_test,training_set=self.train_test_val_split(target,self.conf['split']['test_split'],self.conf['split']['val_split'],self.conf['feature-store']['table_name'],self.conf['feature-store']['lookup_key'],inference_data_df)
        mlflow.set_experiment(self.conf['Mlflow']['experiment_name'])
        with mlflow.start_run() as run:
            # print(self.conf['params'])
            
            model_xgb = xgb.XGBClassifier(**self.conf['params'])

            model_xgb.fit(X_train.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'), y_train)
            y_pred_train = model_xgb.predict(X_train.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_pred_val = model_xgb.predict(X_val.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_pred_test = model_xgb.predict(X_test.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_val_probs = model_xgb.predict_proba(X_val.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_pred_probs = model_xgb.predict_proba(X_test.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            y_train_probs = model_xgb.predict_proba(X_train.drop(self.conf['features']['id_col_list'], axis=1, errors='ignore'))
            
            fpr, tpr, threshold = roc_curve(y_test,y_pred_test)
            roc_auc = auc(fpr, tpr)
            cm=self.confusion_metrics(y_test,y_pred_test)
            fs.log_model(
                                model=model_xgb,
                                artifact_path="usecase",
                                flavor=mlflow.xgboost,
                                training_set=training_set,
                                registered_model_name="usecase_model",
                                )
            
            
            #log all metrics
            mlflow.log_metric("roc_auc",roc_auc)
            
            mlflow.log_metrics(self.metrics(y_train,y_pred_train,y_val,y_pred_val,y_test,y_pred_test))
            self.roc_curve(y_test, y_pred_probs)

            # mlflow.xgboost.log_model(xgb_model=model_xgb,artifact_path="usecase2",registered_model_name="Physician Model")
            mlflow.log_artifact('confusion_matrix.png')
            mlflow.log_artifact('roc_curve.png')
            
            # Save the model as a pickle file
            # with open("model.pkl", "wb") as pickle_file:
            #     pickle.dump(model_xgb, pickle_file)

            # # Log the pickle file as an artifact in MLflow
            # mlflow.log_artifact("model.pkl")
            return X_test,y_test
    def inference(self):
         X_test,y_test=self.train_model()
         spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()
         spark_test = spark.createDataFrame(X_test)

         test_pred = fs.score_batch("models:/usecase_model/latest", spark_test)

         ans_test = test_pred.toPandas()

         y_test = y_test.reset_index()

         y_test.drop('index',axis=1,inplace=True)

         ans_test['actual'] = y_test

         output_df = ans_test[['prediction','actual']]

         print(confusion_matrix(output_df['prediction'],output_df['actual']))

         print(accuracy_score(output_df['prediction'],output_df['actual'])*100)

           

    def launch(self):
        self.logger.info("Launching Model Training task")
        self.inference()
        self.logger.info("Model Training finished!")

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = model_training()

    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()


           