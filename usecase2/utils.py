#function for VIF to check for multi-collinearity
import statsmodels.stats.outliers_influence as sm
import sklearn.feature_selection as sfs
from sklearn.feature_selection import SelectKBest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc,confusion_matrix,classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


def variance_threshold_selection_remove_cols(df:pd.DataFrame, threshold):
  """
  Selects features with a variance greater than a threshold value and returns a list of columns to be removed.

  Args:
    df: The DataFrame to select features from.
    threshold: The minimum variance threshold.

  Returns:
    A list of columns to be removed.
  """

  selector = sfs.VarianceThreshold(threshold=threshold)
  selector = selector.fit(df)

  # cols_to_remove = [col for col in df.columns if col not in selected_features.columns]
  cols_to_remove=[col for col in df.columns 
          if col not in df.columns[selector.get_support()]]


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
  return top_n_col_list



def confusion_metrics(y_test,y_pred,image_path):
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
    plt.savefig(image_path)
    
    return cm,classification_metrics

def roc_curve_fig(y_test, y_prop,image_path):
      """
      Logs Roc_auc curve in MLflow.

      Parameters:
      - y_test: The true labels (ground truth).
      - y_prob: The predicted probabilities of labels (model predictions).
      - run_name: The name for the MLflow run.

      Returns:
      - None
      """
      # y_prop = y_prop[:,1]
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
      # roc_curve_plot_path = imahe
      
      plt.savefig(image_path)

    
      
# def calculate_top_shap_features(df, id_col_list, model, n):
#     """
#         Calculate top  n features.

#         Parameters:
#         - df : dataframe.
#         - id_col_list: lookup key.
#         - model : model classifier
#         - n : no of features

#         Returns:
#         - top n features
#         """
# # Initialize SHAP explainer
#     explainer = shap.Explainer(model)
    
#     # Calculate SHAP values for the entire DataFrame
#     shap_values = explainer.shap_values(df.drop(id_col_list, axis=1))
    
#     # Create a new DataFrame to store the top features for each row
#     top_features_df = pd.DataFrame(index=df.index)
    
#     # Iterate through rows and extract top n features
#     for row_idx in range(len(df)):
#         shap_values_row = shap_values[row_idx]
        
#         # Get the absolute SHAP values
#         abs_shap_values = abs(shap_values_row)
        
#         # Get indices of top n features
#         top_feature_indices = abs_shap_values.argsort()[-n:][::-1]
        
#         # Get corresponding feature names
#         top_feature_names = df.drop(id_col_list, axis=1).columns[top_feature_indices]
        
#         # Add the id_col_list column values to the new DataFrame
#         for col in id_col_list:
#             top_features_df.loc[row_idx, col] = df.loc[row_idx, col]
        
#         # Add the top feature names to the new DataFrame
#         for i in range(n):
#             top_features_df.loc[row_idx, f'REASON{i+1}'] = top_feature_names[i]
    
#     return top_features_df


def push_df_to_s3(df,bucket_name,object_key,s3):
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            s3.Object(bucket_name, object_key).put(Body=csv_content)

            return {"df_push_status": 'successs'}


def preprocess(df_input):
    df_input = df_input.reset_index()
    df_input=df_input.drop('index',axis=1)


    #Clean column names
    print(df_input.columns)
    df_input.columns = df_input.columns.str.strip()
    df_input.columns = df_input.columns.str.replace(' ', '_')
    df_input[self.conf['features']['value_replace']].replace({' M ': 'M', ' F ': 'F'},inplace=True)
    
    df_input.drop(self.conf['features']['drop_col'], axis= 1, inplace= True)
    onehot_cols=self.conf['features']['onehot_cols']
    df_input = pd.get_dummies(df_input, columns=onehot_cols, drop_first=True)
    

    df_feature = df_input.drop(self.conf['features']['target_col'],axis=1)
    return df_input,df_feature
    
  