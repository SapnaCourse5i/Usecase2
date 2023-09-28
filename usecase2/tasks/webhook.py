from mlflow.utils.rest_utils import http_request
import json
def Client():
  return mlflow.tracking.MlflowClient()
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

host_creds = Client()._tracking_client.store.get_host_creds()

def mlflow_call_endpoint(self, endpoint, method, body='{}'):
                if method == 'GET':
                    response = http_request(
                        host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
                else:
                    response = http_request(
                        host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, 
                        json=json.loads(body))
                return response.json()

def webhook():
        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()
        dbutils = DBUtils(spark)
        db_token = dbutils.secrets.get(scope="secrets-scope", key="databricks-token")

        lists = {
            "model_name":"usecase2_model",
            "events": "MODEL_VERSION_TRANSITIONED_TO_PRODUCTION"
        }
        js_list_res = mlflow_call_endpoint('registry-webhooks/list', 'GET', json.dumps(lists))

        print(js_list_res)
        if js_list_res:
              print("Webhook is already created")

        else:
                diction = {
                                "job_spec": {
                                    "job_id": self.conf['deployment-pipeline']['job_id'],
                                    "access_token": db_token,
                                    "workspace_url": self.conf['databricks-url']
                                },
                                "events": [
                                    "MODEL_VERSION_TRANSITIONED_TO_PRODUCTION"
                                ],
                                "model_name": "pharma_model",
                                "description": "Webhook for Deployment Pipeline",
                                "status": "ACTIVE"
                                }

                job_json= json.dumps(diction)
                js_res = self.mlflow_call_endpoint('registry-webhooks/create', 'POST', job_json)
                print(js_res)

                print("Webhook Created for deployment job")
