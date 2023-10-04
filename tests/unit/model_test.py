import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
# from usecase2.tasks.model_pipeline import model_training
# Sample configuration for testing
from conftest import DBUtilsFixture
dbutils=DBUtilsFixture()
from usecase2.tasks.model_pipeline import model_training



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

# # Mocking DBUtils
# @pytest.fixture
# def mock_dbutils():
#     with patch('model_training.DBUtils'):
#         yield

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
#     data = {'A': [1, 2, 3, 4, 5], 'B': [0, 1, 0, 1, 0], 'C': [0, 1, 1, 0, 1],'ID':[1,2,3,3,1],'target':[1,0,1,0,1]}
#     df, X_train, X_val, y_train, y_val, X_test, y_test, training_set = task.train_test_val_split(
#         'target', 0.2, 0.1, 'usecase', 'ID', pd.DataFrame())
#     assert X_train.shape[0] + X_test.shape[0] + X_val.shape[0]== df.shape[0]
# import pytest
# from unittest.mock import MagicMock, patch
# from your_module import train_test_val_split  # Import the function to be tested

# Define a sample configuration for testing
sample_config = {
    'Mlflow': {
        'experiment_name': 'test_experiment'
    },
    # Add other sample configurations here
}

# Define a sample DataFrame for inference_data_df
sample_inference_data = pd.DataFrame({
    'lookup_key': [1, 2, 3, 4, 5],
    'target': [0, 1, 0, 1, 0]
})

@pytest.fixture
def mock_feature_lookup():
    # Mock the FeatureLookup class
    with patch('model_training.FeatureLookup', autospec=True) as mock_feature_lookup_class:
        # Mock the behavior of the FeatureLookup instance
        mock_instance = mock_feature_lookup_class.return_value
        mock_instance.load_df.return_value.toPandas.return_value = sample_inference_data
        yield mock_feature_lookup_class

def test_train_test_val_split(mock_feature_lookup):
    # Define test parameters
    target = 'target'
    test_split = 0.2
    val_split = 0.1
    table_name = 'usecase2'
    lookup_key = 'lookup_key'
    
    # Call the function with mock FeatureLookup
    df, X_train, X_val, y_train, y_val, X_test, y_test, training_set = model_training.train_test_val_split(
        target, test_split, val_split, table_name, lookup_key, sample_inference_data
    )
    
    # Check that FeatureLookup was instantiated with the correct parameters
    mock_feature_lookup.assert_called_once_with(table_name=table_name, lookup_key=lookup_key)

    # Check that create_training_set was called with the correct parameters
    mock_feature_lookup_instance = mock_feature_lookup.return_value
    mock_feature_lookup_instance.create_training_set.assert_called_once_with(
        sample_inference_data, [mock_feature_lookup_instance], label=target, exclude_columns=lookup_key
    )

    # Check the return values
    assert isinstance(df, pd.DataFrame)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(training_set, MagicMock)  # You can customize this check based on the actual type of training_set

    # Perform additional checks as needed for the returned DataFrames and Series

    # Example checks:
    assert df.shape == (5, 2)
    assert X_train.shape == (3, 1)
    assert X_val.shape == (1, 1)
    assert y_train.shape == (3,)
    assert y_val.shape == (1,)
    assert X_test.shape == (1, 1)
    assert y_test.shape == (1,)

# Run the test using pytest

    
    # Add assertions based on your expected behavior

# def test_metrics():
#     task = model_training()
#     # Create sample data
#     y_train = [1, 0, 1, 1, 0, 0]
#     y_pred_train = [1, 0, 1, 0, 1, 0]
#     y_val = [1, 0, 1, 1, 0, 0]
#     y_pred_val = [1, 0, 1, 0, 1, 0]
#     y_test = [1, 0, 1, 1, 0, 0]
#     y_pred_test = [1, 0, 1, 0, 1, 0]

#     metrics_result = task.metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
#     assert 'accuracy_train' in metrics_result
#     assert 'accuracy_val' in metrics_result
#     assert 'accuracy_test' in metrics_result
#     assert 'f1 score train' in metrics_result
#     assert 'f1 score val' in metrics_result
#     assert 'f1 score test' in metrics_result
    
#     # Check if the metrics are within the valid range (0 <= value <= 1)
#     for key, value in metrics_result.items():
#         assert 0 <= value <= 1
    
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

