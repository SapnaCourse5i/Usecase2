import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from usecase2.tasks.model_pipeline import model_training
# Sample configuration for testing
from conftest import DBUtilsFixture


# In your test file, before importing the class with 'pyspark.dbutils'
# from unittest.mock import MagicMock, patch

# Mock the 'pyspark.dbutils' import
# with patch('usecase2.tasks.model_pipeline.DBUtils', MagicMock()):
#     # Import the class that uses 'pyspark.dbutils'
#     from usecase2.tasks.model_pipeline import model_training

# Rest of your test code

sample_config = {
    'Mlflow': {
        'experiment_name': 'test_experiment'
    },
    # Add other sample configurations here
}

# Mocking mlflow.start_run and mlflow.end_run
@pytest.fixture
def mock_mlflow():
    with patch('model_training.mlflow.start_run'), patch('model_training.mlflow.end_run'):
        yield

# Mocking DBUtils
@pytest.fixture
def mock_dbutils():
    with patch('model_training.DBUtils'):
        yield

# Mocking s3.Object and pandas.read_csv
@pytest.fixture
def mock_s3_and_read_csv():
    with patch('model_training.s3.Object') as mock_s3_object, \
         patch('model_training.pd.read_csv') as mock_read_csv:
        mock_s3_object.return_value.get.return_value = {'Body': MagicMock()}
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        yield

# def test_train_test_val_split(mock_dbutils, mock_s3_and_read_csv):
#     task = model_training()
#     task.conf = sample_config  # Set the sample configuration
#     df, X_train, X_val, y_train, y_val, X_test, y_test, training_set = task.train_test_val_split(
#         'target', 0.2, 0.1, 'table_name', 'lookup_key', pd.DataFrame())
#     assert X_train.shape[0] + X_test.shape[0]== df.shape[0]
    
    # Add assertions based on your expected behavior

def test_metrics():
    task = model_training()
    # Create sample data
    y_train = [1, 0, 1, 1, 0, 0]
    y_pred_train = [1, 0, 1, 0, 1, 0]
    y_val = [1, 0, 1, 1, 0, 0]
    y_pred_val = [1, 0, 1, 0, 1, 0]
    y_test = [1, 0, 1, 1, 0, 0]
    y_pred_test = [1, 0, 1, 0, 1, 0]

    metrics_result = task.metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
    assert 'accuracy_train' in metrics_result
    assert 'accuracy_val' in metrics_result
    assert 'accuracy_test' in metrics_result
    assert 'f1 score train' in metrics_result
    assert 'f1 score val' in metrics_result
    assert 'f1 score test' in metrics_result
    
    # Check if the metrics are within the valid range (0 <= value <= 1)
    for key, value in metrics_result.items():
        assert 0 <= value <= 1
    
    # Add assertions based on your expected behavior

# def test_train_model(mock_mlflow, mock_dbutils, mock_s3_and_read_csv):
#     task = model_training()
#     task.conf = sample_config  # Set the sample configuration
#     task.train_model()

    # Add assertions based on your expected behavior

# def test_inference(mock_mlflow):
#     task = model_training()
#     task.conf = sample_config  # Set the sample configuration
#     task.inference()

    # Add assertions based on your expected behavior

# Add more test cases for other methods as needed

