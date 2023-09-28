from usecase2.common import Task

import json
import pandas as pd
import requests
import zipfile
import io

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# from evidently.pipeline.column_mapping import ColumnMapping
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from mlflow.utils.rest_utils import http_request
import json
def client():
  return mlflow.tracking.MlflowClient()
 
host_creds = client()._tracking_client.store.get_host_creds()


def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, 
          json=json.loads(body))
  return response.json()

class Webhook(Task):


    def _create_webhook(self):


        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        dbutils = DBUtils(spark)

        aws_access_key = dbutils.secrets.get(scope="secrets-scope", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope", key="aws-secret-key")
        db_host = dbutils.secrets.get(scope="secrets-scope", key="databricks-host")
        db_token = dbutils.secrets.get(scope="secrets-scope", key="databricks-token")

        lists = {
            "model_name":"usecase2_model",
            "events": "MODEL_VERSION_TRANSITIONED_TO_PRODUCTION"
        }
        js_list_res = mlflow_call_endpoint('registry-webhooks/list', 'GET', json.dumps(lists))

        if js_list_res:
              print("Webhook is already created")

        else:
                diction = {
                                "job_spec": {
                                    "job_id": self.conf['deployment-pipeline']['job_id'],
                                    "access_token": db_token,
                                    "workspace_url": db_host
                                },
                                "events": [
                                    "MODEL_VERSION_TRANSITIONED_TO_PRODUCTION"
                                ],
                                "model_name": "usecase2_model",
                                "description": "Webhook for Deployment Pipeline",
                                "status": "ACTIVE"
                                }

                job_json= json.dumps(diction)
                js_res = mlflow_call_endpoint('registry-webhooks/create', 'POST', job_json)
                print(js_res)

                print("Webhook Created for deployment job")

       


              
              

    def launch(self):
         
         self._create_webhook()

   
def entrypoint():  
    
    task = Webhook()
    task.launch()

if __name__ == '__main__':
    entrypoint()