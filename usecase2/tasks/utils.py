#function for VIF to check for multi-collinearity
import statsmodels.stats.outliers_influence as sm
import sklearn.feature_selection as sfs
from sklearn.feature_selection import SelectKBest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc


def variance_threshold_selection_remove_cols(df, threshold):
  """
  Selects features with a variance greater than a threshold value and returns a list of columns to be removed.

  Args:
    df: The DataFrame to select features from.
    threshold: The minimum variance threshold.

  Returns:
    A list of columns to be removed.
  """

  selector = sfs.VarianceThreshold(threshold=threshold)
  selected_features = selector.fit_transform(df)

  cols_to_remove = [col for col in df.columns if col not in selected_features.columns]

  return cols_to_remove

def select_kbest_features(df, target_col,n):
  """
  Selects the top n features from the DataFrame using the SelectKBest algorithm.

  Args:
    df: The DataFrame to select features from.
    n: The number of features to select.

  Returns:
    A list of the top n features.
  """


  selector = SelectKBest(k=n)
  selected_features = selector.fit_transform(df, target_col)
  
  mask = selector.get_support()
  top_n_features = df.columns[mask]
  top_n_col_list = top_n_features.tolist()


  # selector = SelectKBest(k=self.conf['kbestfeatures']['no_of_features'])
  # df_input=self.preprocessing()
  # target_col = df_input[self.conf['features']['target_col']]
  # id_col_list = self.conf['features']['id_col_list']
  # df_input1=df_input.drop(id_col_list,axis=1)
  # selected_features = selector.fit_transform(df_input1, target_col)

  # mask = selector.get_support()
  # top_n_features = df_input1.columns[mask]
  # top_n_col_list = top_n_features.tolist()
  
  # cols_for_model_df_list = id_col_list + top_n_col_list
  # df_final=df_input[cols_for_model_df_list]
  # df_final[id_col_list]=df_input[id_col_list]
  # return top_n_col_list
  return top_n_col_list

def apply_model(model, X_train, y_train, X_val, y_val, drop_id_col_list):
    # Fit the model
    model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)

    # Make predictions
    y_train_pred = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
    y_pred = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

    # Calculate performance metrics
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_pred)
    f1_train = f1_score(y_train,y_train_pred)
    f1_val = f1_score(y_val, y_pred)
    return accuracy_train, accuracy_val,f1_train,f1_val
  