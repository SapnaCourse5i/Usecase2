import pandas as pd
import numpy as np
import warnings

from usecase2.common import Task
#Visual Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pickle
import shap
import io
from pyspark.sql import udf
# from pyspark.sql.types import ArrayType, StructField, StructType, StringType , IntegerType

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc,classification_report
import xgboost as xgb
# from evidently.pipeline.column_mapping import ColumnMapping
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

from databricks import feature_store
from pyspark.sql import SparkSession
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from databricks.feature_store import feature_table, FeatureLookup
from usecase2.utils import select_kbest_features
# from utils import select_kbest_features

# from evidently import ColumnMapping

# from evidently.metric_preset import DataDriftPreset
# from utils import apply_model

warnings.filterwarnings('ignore')
from pyspark.dbutils import DBUtils

fs = feature_store.FeatureStoreClient()


class model_training(Task):

    def train_test_val_split(self,target,test_split,val_split,table_name,lookup_key,inference_data_df):
        """
            Split data into train,test,validation

            Parameters:
            - target: The true labels (ground truth).
            - test_split: test split ratio.
            - val_split: validation split ratio.
            - table name - feature store table name
            - lookup key - primary key
            - inference_data_df - dataframe with lookup key and target column

            Returns:
            - X_train , X_test , X_val , y_train , y_test , y_val 
        """


        model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
                
        # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
        training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label=target,exclude_columns=lookup_key)
        df= training_set.load_df().toPandas()
        # df1=df.drop(self.conf['features']['target_col'],axis=1)
        # print(df1.columns)
        # top_features=select_kbest_features(df1,df[self.conf['features']['target_col']],n=self.conf['kbestfeatures']['no_of_features'])
        # df=df[top_features+ [self.conf['features']['target_col']]]
        print(df.columns)
        # X_train, X_val, y_train, y_val,X_test,y_test=self.train_test_val_split(df_input,test_split,val_split)
        

        X = df.drop(target, axis=1)
        y = df[target]

        # Performing the train-test split
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify= y)

        # Performing the train-val split using train data
        X_train, X_val, y_train, y_val = train_test_split(X_train_pre, y_train_pre, test_size=val_split, random_state=42, stratify= y_train_pre)
        return df,X_train, X_val, y_train, y_val,X_test,y_test,training_set#,top_features
    
    def metrics(self,y_train,y_pred_train,y_val,y_pred_val,y_test,y_pred):
        """
            Logs f1_Score and accuracy in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            - y_pred: The predicted labels (model predictions).
            - run_name: The name for the MLflow run.

            Returns:
            - f1score and accuracy
        """

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

    
      
    def calculate_top_shap_features(df, id_col_list, model, n):
        """
            Calculate top  n features.

            Parameters:
            - df : dataframe.
            - id_col_list: lookup key.
            - model : model classifier
            - n : no of features

            Returns:
            - top n features
            """
    # Initialize SHAP explainer
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values for the entire DataFrame
        shap_values = explainer.shap_values(df.drop(id_col_list, axis=1))
        
        # Create a new DataFrame to store the top features for each row
        top_features_df = pd.DataFrame(index=df.index)
        
        # Iterate through rows and extract top n features
        for row_idx in range(len(df)):
            shap_values_row = shap_values[row_idx]
            
            # Get the absolute SHAP values
            abs_shap_values = abs(shap_values_row)
            
            # Get indices of top n features
            top_feature_indices = abs_shap_values.argsort()[-n:][::-1]
            
            # Get corresponding feature names
            top_feature_names = df.drop(id_col_list, axis=1).columns[top_feature_indices]
            
            # Add the id_col_list column values to the new DataFrame
            for col in id_col_list:
                top_features_df.loc[row_idx, col] = df.loc[row_idx, col]
            
            # Add the top feature names to the new DataFrame
            for i in range(n):
                top_features_df.loc[row_idx, f'REASON{i+1}'] = top_feature_names[i]
        
        return top_features_df
    
    
        

    
    def train_model(self):


        """
            Train model on Xgboost 
            Evaluate the model
            Log the model using mlflow
            Inference the model.

        """
        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        dbutils = DBUtils(spark)

        aws_access_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-secret-key")
        
        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                aws_secret_access_key=aws_secret_key, 
                region_name='ap-south-1')
        
        bucket_name =  self.conf['s3']['bucket_name']
        csv_file_key = self.conf['cleaned_data']['final_features_df_path']
        # print('start')

        
        s3_object = s3.Object(bucket_name, csv_file_key)
        
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))

        

        df_input_spark = spark.createDataFrame(df_input)

        inference_data_df = df_input_spark.select(self.conf['features']['id_target_col_list'])



        target=self.conf['features']['target_col']
    

        df,X_train, X_val, y_train, y_val,X_test,y_test,training_set=self.train_test_val_split(target,self.conf['split']['test_split'],self.conf['split']['val_split'],self.conf['feature-store']['table_name'],self.conf['feature-store']['lookup_key'],inference_data_df)
        # df_train_spark = spark.createDataFrame(X_train.drop(self.conf['features']['id_col_list'],axis=1))

        # csv_buffer = BytesIO()
        # X_test.to_csv(csv_buffer, index=False)
        # csv_content = csv_buffer.getvalue()

        # s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
        #             aws_secret_access_key=aws_secret_key, 
        #             region_name='ap-south-1')

        # s3_object_key = self.conf['s3']['X_test'] 
        # s3.Object(self.conf['s3']['bucket_name'], s3_object_key).put(Body=csv_content)
        # print('uploaded file')
        mlflow.set_experiment(self.conf['Mlflow']['experiment_name'])
        with mlflow.start_run() as run:
            # print(self.conf['params'])
            
            # model_xgb = xgb.XGBClassifier()
            model_xgb=LogisticRegression()
            id_col_list=self.conf['features']['id_col_list']

            model_xgb.fit(X_train.drop(id_col_list, axis=1, errors='ignore'), y_train)
            y_pred_train = model_xgb.predict(X_train.drop(id_col_list, axis=1, errors='ignore'))
            y_pred_val = model_xgb.predict(X_val.drop(id_col_list, axis=1, errors='ignore'))
            y_pred_test = model_xgb.predict(X_test.drop(id_col_list, axis=1, errors='ignore'))
            y_val_probs = model_xgb.predict_proba(X_val.drop(id_col_list, axis=1, errors='ignore'))
            y_pred_probs = model_xgb.predict_proba(X_test.drop(id_col_list, axis=1, errors='ignore'))
            y_train_probs = model_xgb.predict_proba(X_train.drop(id_col_list, axis=1, errors='ignore'))
            print(y_pred_test)
            print(y_test)
            fpr, tpr, threshold = roc_curve(y_test,y_pred_test)
            roc_auc = auc(fpr, tpr)
            cm=self.confusion_metrics(y_test,y_pred_test)
            fs.log_model(
                                model=model_xgb,
                                artifact_path="usecase",
                                flavor=mlflow.xgboost,
                                training_set= training_set,
                                registered_model_name="usecase2_model",
                                )
            
            
            #log all metrics
            mlflow.log_metric("roc_auc",roc_auc)
            
            mlflow.log_metrics(self.metrics(y_train,y_pred_train,y_val,y_pred_val,y_test,y_pred_test))
            self.roc_curve(y_test, y_pred_probs)

            # mlflow.xgboost.log_model(xgb_model=model_xgb,artifact_path="usecase2",registered_model_name="Physician Model")
            mlflow.log_artifact('confusion_matrix.png')
            mlflow.log_artifact('roc_curve.png')
            
            # Save the model as a pickle file
            with open("model.pkl", "wb") as pickle_file:
                pickle.dump(model_xgb, pickle_file)

            # Log the pickle file as an artifact in MLflow
            mlflow.log_artifact("model.pkl")

            #  Create a SHAP explanation
            # explainer = shap.Explainer(model_xgb, X_val.drop(id_col_list,axis=1))
            explainer = shap.Explainer(model_xgb, X_val)
            # shap_values = explainer(X_test.drop(id_col_list,axis=1))
            shap_values = explainer(X_test)
            # Visualize the SHAP explanation
            # shap.plots.bar(shap_values[1],show=False)
            # shap.summary_plot(shap_values, X_test.drop(id_col_list,axis=1),show=False)
            shap.summary_plot(shap_values, X_test,show=False)
            plt.savefig('summary_plot.png')
            mlflow.log_artifact('summary_plot.png')


            

        return X_test
    #,y_test,X_val,df_input_spark.select(self.conf['features']['id_col_list'])
    

    def inference(self):
         X_test=self.train_model()
        #  print(X_test.shape)
        #  print(X_test.columns)
        #  print(y_test)
      
         X_test1=fs.read_table(self.conf['feature-store']['table_name'])
         
        #  spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()
        #  spark_test = spark.createDataFrame(X_test)
        # #  batch_df=X_test[self.conf['features']['id_col_list']]
        #  print(len(X_test1.columns))
        #  print(X_test1.count())
        # #  inference_list=X_test['NPI_ID'].tolist()
        #  all_features =top_features+ self.conf['features']['id_col_list']
        #  print(all_features)
        #  X_test2=X_test1.select(all_features)
        #  print(len(X_test2.columns))

        #  X_test1=X_test1.filter(X_test1['NPI_ID'].isin(spark_test))
        #  print(len(X_test1.columns))
         
         print('scoring now')
         test_pred = fs.score_batch("models:/usecase2_model/latest", X_test1)
         print('scoring done')
         print(len(test_pred.columns))
         print(test_pred.count())


         ans_test = test_pred.toPandas()

         print('converted to pandas')
         print(ans_test.columns)
        #  X_test1=X_test1.toPandas()
        #  df_refernce=X_test1[:2500,:]
        #  df_test=X_test1[2500:,:]
        #  y_pred=model_xgb.predict(X_test.drop(self.conf['features']['id_col_list'],axis=1))
        #  y_test = y_test.reset_index()
        #  appended_df = test_pred.union(y_test)
        #  print(appended_df.columns)
        #  y_test=y_test.tolist()
        #  def add_labels(indx):
        #     return y_test[indx-1] # since row num begins from 1
        #  labels_udf = udf(add_labels, IntegerType())

        # #  a = spark.createDataFrame([("Dog", "Cat"), ("Cat", "Dog"), ("Mouse", "Cat")],["Animal", "Enemy"])
        #  test_pred.createOrReplaceTempView('a')
        #  a = spark.sql('select row_number() over (order by "Animal") as num, * from a')

        #  a.show()
        #  y_pred = test_pred.select(['prediction']).collect()
        #  y_pred = predictions_train.select(['prediction']).collect()


        #  print(classification_report(y_test, y_pred))
        #  df_pred = test_pred.withColumn("actual", col("col2") * numpy_array)

        #  ans_test = test_pred.select('prediction')
        #  print(ans_test)
        #  ans_test = ans_test.toPandas()

        #  print('created test')

        #  y_test = y_test.reset_index()

        #  y_test.drop('index',axis=1,inplace=True)

        #  ans_test['actual'] = y_test

        #  output_df = ans_test[['prediction','actual']]

        #  print(confusion_matrix(output_df['prediction'],output_df['actual']))

        #  print(accuracy_score(output_df['prediction'],output_df['actual'])*100)

        #  top_features_df=self.calculate_top_shap_features(X_test,self.conf['features']['id_Col_list'],model=model,n=3)
        #  print(top_features_df.head())

        

         # Visualize the SHAP explanation
        #  shap.plots.bar(shap_values[1],show=False)
        #  shap.summary_plot(shap_values, X_test,show=False)
        #  fig1=plt.savefig('summary_plot.png')


         
        # Save the output to a Bytes IO object
        #  png_data = io.BytesIO()
        #  fig1.savefig(png_data)
        # Seek back to the start so boto3 uploads from the start of the data
        #  bits.seek(0)

        # Upload the data to S3
        #  s3 = boto3.client('s3')
         
        #  s3.put_object(Bucket=[self.conf['s3']['bucket_name']], Key=self.conf['s3']['figure_path'], Body=png_data)


           

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


           