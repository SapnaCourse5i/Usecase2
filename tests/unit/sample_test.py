import logging
from pathlib import Path

import mlflow
from pyspark.sql import SparkSession
from unittest.mock import MagicMock
# warnings.filterwarnings('ignore')
# from pyspark.dbutils import DBUtils
# from databricks.feature_store import feature_table, FeatureLookup,FeatureStoreClient
# fs = FeatureStoreClient()

from conftest import DBUtilsFixture
# from usecase2.tasks.sample_etl_task import SampleETLTask
# from usecase2.tasks.sample_ml_task import SampleMLTask


# def test_jobs(spark: SparkSession, tmp_path: Path):
#     logging.info("Testing the ETL job")
#     common_config = {"database": "default", "table": "sklearn_housing"}
#     test_etl_config = {"output": common_config}
#     etl_job = SampleETLTask(spark, test_etl_config)
#     etl_job.launch()
#     table_name = f"{test_etl_config['output']['database']}.{test_etl_config['output']['table']}"
#     _count = spark.table(table_name).count()
#     assert _count > 0
#     logging.info("Testing the ETL job - done")

#     logging.info("Testing the ML job")
#     test_ml_config = {
#         "input": common_config,
#         "experiment": "/Shared/UseCase2/sample_experiment"
#     }
#     ml_job = SampleMLTask(spark, test_ml_config)
#     ml_job.launch()
#     experiment = mlflow.get_experiment_by_name(test_ml_config['experiment'])
#     assert experiment is not None
#     runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
#     assert runs.empty is False
#     logging.info("Testing the ML job - done")

import pandas as pd
import numpy as np
import mlflow
import os
import shutil
import tempfile
import pytest
from io import BytesIO
from unittest.mock import patch, Mock
from  usecase2.utils import (
    variance_threshold_selection_remove_cols,
    select_kbest_features,
    confusion_metrics,
    roc_curve_fig,
    # calculate_top_shap_features,
    push_df_to_s3,

)
from usecase2.tasks.model_pipeline import model_training


# Fixture to create a temporary MLflow run
@pytest.fixture(scope="function")
def mlflow_run():
    mlflow.start_run()
    yield
    mlflow.end_run()

# Test variance_threshold_selection_remove_cols function
def test_variance_threshold_selection_remove_cols():
    # Create a sample DataFrame
    data = {'A': [1, 2, 3, 4, 5], 'B': [0, 0, 0, 0, 0], 'C': [1, 1, 1, 1, 1]}
    df = pd.DataFrame(data)
    
    # Set the threshold
    threshold = 0.1
    
    # Test the function
    cols_to_remove = variance_threshold_selection_remove_cols(df, threshold)
    
    # Check the expected result
    assert cols_to_remove == ['B','C']

# Test select_kbest_features function
def test_select_kbest_features():
    # Create a sample DataFrame
    data = {'A': [1, 2, 3, 4, 5], 'B': [0, 1, 0, 1, 0], 'C': [0, 1, 1, 0, 1]}
    df = pd.DataFrame(data)
    target_col = df['B']
    n = 2
    
    # Test the function
    top_n_features = select_kbest_features(df, target_col, n)
    
    # Check the expected result
    assert len(top_n_features) == n

# Test confusion_metrics function
# def test_confusion_metrics(mlflow_run):
#     # Create sample data
#     y_test = [1, 0, 1, 1, 0, 0]
#     y_pred = [1, 0, 1, 0, 1, 0]
    
#     # Call the function
#     cm, classification_metrics = confusion_metrics(y_test, y_pred)
#     # Create a MagicMock for mlflow.log_figure
#     mlflow.log_figure = MagicMock()
#     # Check the shape of the confusion matrix
#     assert cm.shape == (2, 2)

        
#     # Check that mlflow.log_figure was called during the run
#     mlflow.log_figure.assert_called()
def test_confusion_metrics(mlflow_run):
    # Create sample data
    y_test = [1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    
    # Specify the path where the image will be saved
    image_path = 'confusion_matrix.png'
    
    # Call the function
    confusion_metrics(y_test, y_pred, image_path)
    
    # Check if the file exists
    assert os.path.isfile(image_path)

# Test roc_curve function
def test_roc_curve(mlflow_run):
    # Create sample data
    y_test = [1, 0, 1, 1, 0, 0]
    y_prob = [0.8, 0.2, 0.6, 0.7, 0.3, 0.4]
    image_path = 'roc_auc_curve.png'
    # Call the function
    roc_curve_fig(y_test, y_prob,image_path)
    # mlflow.log_figure = MagicMock()
    
    # Check that mlflow.log_figure was called during the run
    assert os.path.isfile(image_path)

# Test calculate_top_shap_features function
# def test_calculate_top_shap_features():
#     # Create sample data
#     data = {'ID': [1, 2, 3, 4, 5], 'B': [0, 1, 0, 1, 0], 'C': [0, 1, 1, 0, 1],'D':[2, 4,0,1,1]}
#     df = pd.DataFrame(data)
#     id_col_list = ['ID']
#     model = Mock()
#     n = 2
    
#     # Call the function
#     top_features_df = calculate_top_shap_features(df, id_col_list, model, n)
    
#     # Check the shape of the resulting DataFrame
#     assert top_features_df.shape == (5, 4)

# Test push_df_to_s3 function
def test_push_df_to_s3():
    # Create sample data
    data = {'A': [1, 2, 3, 4, 5], 'B': [0, 1, 0, 1, 0]}
    df = pd.DataFrame(data)
    bucket_name = 's3'
    object_key = 'test-data.csv'
    
    # Mock the S3 object
    s3_mock = Mock()
    
    # Call the function
    result = push_df_to_s3(df, bucket_name, object_key, s3_mock)
    
    # Check the result
    assert result == {"df_push_status": 'successs'}
    s3_mock.Object.assert_called_once_with(bucket_name, object_key)
    s3_mock.Object().put.assert_called_once()




# def test_metrics():
    # # Create sample data
    # y_train = [1, 0, 1, 1, 0, 0]
    # y_pred_train = [1, 0, 1, 0, 1, 0]
    
    # y_val = [1, 0, 1, 1, 0, 0]
    # y_pred_val = [1, 0, 1, 0, 1, 0]
    
    # y_test = [1, 0, 1, 1, 0, 0]
    # y_pred_test = [1, 0, 1, 0, 1, 0]
    
    # # Call the metrics function
    # metrics_result = ms.metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
    
    # # Check if the metrics are within the expected range
    # assert 'accuracy_train' in metrics_result
    # assert 'accuracy_val' in metrics_result
    # assert 'accuracy_test' in metrics_result
    # assert 'f1 score train' in metrics_result
    # assert 'f1 score val' in metrics_result
    # assert 'f1 score test' in metrics_result
    
    # # Check if the metrics are within the valid range (0 <= value <= 1)
    # for key, value in metrics_result.items():
    #     assert 0 <= value <= 1




