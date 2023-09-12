import pandas as pd
# from sklearn.datasets import fetch_california_housing
from usecase2.common import Task
import boto3
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import warnings
import os
import urllib
import pickle
from pyspark.sql import SparkSession
from io import BytesIO
import uuid
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from databricks import feature_store

from sklearn.model_selection import train_test_split

from databricks.feature_store import feature_table, FeatureLookup

import os

from pyspark.dbutils import DBUtils


class FeatureEngineering_Pipeline(Task):
    def push_df_to_s3(self,df,access_key,secret_key):
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            s3 = boto3.resource("s3",aws_access_key_id=access_key, 
                      aws_secret_access_key=secret_key, 
                      region_name='ap-south-1')

            s3_object_key = self.conf['cleaned_data']['preprocessed_df_path'] 
            s3.Object(self.conf['s3']['bucket_name'], s3_object_key).put(Body=csv_content)

            return {"df_push_status": 'success'}

    def push_final_feature_df_to_s3(self,df,access_key,secret_key):
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            s3 = boto3.resource("s3",aws_access_key_id=access_key, 
                      aws_secret_access_key=secret_key, 
                      region_name='ap-south-1')

            s3_object_key = self.conf['cleaned_data']['features/final_features_df.csv'] 
            s3.Object(self.conf['s3']['bucket_name'], s3_object_key).put(Body=csv_content)

            return {"df_push_status": 'success'}
    def preprocessing(self):
        # db = self.conf["output"].get("database", "default")
        # table = self.conf["output"]["table"]
        # self.logger.info(f"Writing housing dataset to {db}.{table}")
        # _data: pd.DataFrame = fetch_california_housing(as_frame=True).frame
        # df = self.spark.createDataFrame(_data)
        # df.write.format("delta").mode("overwrite").saveAsTable(f"{db}.{table}")
        # self.logger.info("Dataset successfully written")
        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        dbutils = DBUtils(spark)

        aws_access_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-secret-key")
        
        
        access_key = aws_access_key 
        secret_key = aws_secret_key

        

                
                
                # encoded_secret_key = urllib.parse.quote(secret_key,safe="")
        
        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name='ap-south-1')
                
        bucket_name =  self.conf['s3']['bucket_name']
        csv_file_key = self.conf['s3']['filepath']

        s3_object = s3.Object(bucket_name, csv_file_key)
        
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))
        df_input = df_input.reset_index()
 
        #Clean column names
       
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')
        df_input['Sex'].replace({' M ': 'M', ' F ': 'F'},inplace=True)
        
        df_input.drop(self.conf['features']['drop_col'], axis= 1, inplace= True)
        onehot_cols=self.conf['features']['onehot_cols']
        df_input = pd.get_dummies(df_input, columns=onehot_cols, drop_first=True)
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.conf['feature-store']['table_name']}")
                # Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
        table_name = self.conf['feature-store']['table_name']
        print(table_name)

        df_feature = df_input.drop(self.conf['features']['target_col'],axis=1)

        df_spark = spark.createDataFrame(df_feature)

        fs = feature_store.FeatureStoreClient()

        fs.create_table(
                name=table_name,
                primary_keys=[self.conf['feature-store']['lookup_key']],
                df=df_spark,
                schema=df_spark.schema,
                description="health features"
            )
        
        push_status = self.push_df_to_s3(df_input,access_key,secret_key)
        print(push_status)
        return df_input
        
    
    def feature_selection(self):
        #   col_for_feature_selection = df_input.columns.difference(self.conf['features']['id_target_col_list'])
    #     var_thr = VarianceThreshold(threshold = 0.1) #Removing both constant and quasi-constant
    #     var_thr.fit(df_input[col_for_feature_selection])

    #     var_thr.get_support()
          selector = SelectKBest(k=self.conf['kbestfeatures']['no_of_features'])
          df_input=self.preprocessing()
          target_col = df_input[self.conf['features']['target_col']]
          selected_features = selector.fit_transform(df_input, target_col)
        
          mask = selector.get_support()
          top_n_features = df_input.columns[mask]
          id_col_list = self.conf['features']['id_col_list']
          cols_for_model_df_list = id_col_list + top_n_features
          df_final=df_input[cols_for_model_df_list]
          spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

          dbutils = DBUtils(spark)

          aws_access_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-access-key")
          aws_secret_key = dbutils.secrets.get(scope="secrets-scope2", key="aws-secret-key")
          access_key = aws_access_key 
          secret_key = aws_secret_key
          push_status = self.push_final_feature_df_to_s3(df_final,access_key,secret_key)
          print(push_status)
        #   top_n_col_list = select_kbest_features(df_input.drop(id_col_list,axis=1),target_col, 30)
          
       
            
   

    def launch(self):
        self.logger.info("Launching sample ETL task")
        self.feature_selection()
        self.logger.info("Sample ETL task finished!")

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = FeatureEngineering_Pipeline()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
