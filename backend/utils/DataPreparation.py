import numpy as np
from sklearn.decomposition import IncrementalPCA
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

def drop_duplicates(data):
    """
        This function removes any duplicate entries from the provided Pandas DataFrame.
        Duplicates are determined based on all columns.

        Parameters:
        data (pandas.DataFrame): The DataFrame from which duplicates will be removed.

        Returns:
        pandas.DataFrame: The DataFrame after removal of duplicates.
    """

    data.drop_duplicates(inplace=True)
    return data

def deal_with_missing_values(data):
    """
        This function processes a Pandas DataFrame to replace infinite values
        (both positive and negative) with NaN (not a number), and then fill NaN
        values in the 'Flow Bytes/s' and 'Flow Packets/s' columns with the median
        value of those respective columns.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed.

        Returns:
        pd.DataFrame: The processed DataFrame with handled missing values.
    """

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['Flow Bytes/s'].fillna(data['Flow Bytes/s'].median(), inplace=True)
    data['Flow Packets/s'].fillna(data['Flow Packets/s'].median(), inplace=True)
    return data

def z_score_normalization(data):
    """
       This function normalizes the numeric features in the provided Pandas DataFrame using Z-score normalization.
       Z-score normalization is an approach of normalizing data that avoids issues of scale differences between features.
       The function only applies to columns that do not contain object data types.

       Parameters:
       data (pandas.DataFrame): DataFrame containing the features to be normalized.

       Returns:
       pandas.DataFrame: The DataFrame with normalized numeric features.
    """

    features = data.dtypes[data.dtypes != 'object'].index
    data[features] = data[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    return data


def preprocessing(data):
    """
        This function performs several preprocessing steps on the provided Pandas DataFrame:
        - Removes duplicate entries
        - Replaces infinite values with NaN
        - Fills missing values in the 'Flow Bytes/s' and 'Flow Packets/s' columns with the column median
        - Applies Z-score normalization to numeric features
        - Maps labels in the 'Label' column to broader attack types
        - Encodes the 'Label' column using LabelEncoder
        - Removes features with only one unique value.

        Note that some operations are performed in-place and modify the original DataFrame.

        Parameters:
        data (pandas.DataFrame): The DataFrame to be preprocessed.

        Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    # Dropping duplicates
    data.drop_duplicates(inplace=True)

    # Replacing any infinite values (positive or negative) with NaN (not a number)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filling missing values with median
    data['Flow Bytes/s'].fillna(data['Flow Bytes/s'].median(), inplace=True)
    data['Flow Packets/s'].fillna(data['Flow Packets/s'].median(), inplace=True)

    # Z-score normalization
    features = data.dtypes[data.dtypes != 'object'].index
    data[features] = data[features].apply(
        lambda x: (x - x.mean()) / (x.std()))


    # dropping features with one unique value
    num_unique = data.nunique()
    not_one_variable = num_unique[num_unique > 1].index
    data = data[not_one_variable]

    return data


def feature_engineering(data, n_features):
    """
        This function performs feature engineering on the provided Pandas DataFrame.
        It performs following tasks:
        - Splits the input data into training and testing sets for feature importance calculation.
        - Calculates the importance of features using the mutual information metric.
        - Selects important features till the accumulated importance reaches 90%.
        - Uses Incremental Principal Component Analysis (IPCA) for dimensionality reduction while preserving
          99% of the variance.
        - Transforms the features using the identified PCA components.
        - In the end, it creates a new DataFrame with transformed features and original labels.

        The 'Label' column is dropped from the features and is assumed to be the last column of the DataFrame
        for labels.

        Parameters:
        data (pandas.DataFrame): Original DataFrame to be processed.

        Returns:
        pandas.DataFrame: New DataFrame with transformed features and intact labels.
    """
    columns_selected = ['Average Packet Size', 'Packet Length Variance',
                        'Packet Length Std', 'Packet Length Mean',
                        'Total Length of Bwd Packets', 'Subflow Bwd Bytes',
                        'Bwd Packet Length Mean', 'Avg Bwd Segment Size',
                        'Init_Win_bytes_backward', 'Destination Port',
                        'Total Length of Fwd Packets', 'Bwd Packet Length Max',
                        'Subflow Fwd Bytes', 'Max Packet Length', 'Fwd Packet Length Max',
                        'Init_Win_bytes_forward', 'Fwd IAT Max', 'Fwd IAT Total',
                        'Flow IAT Max', 'Fwd IAT Mean', 'Fwd Header Length.1',
                        'Fwd Header Length', 'Flow Duration', 'Bwd Packets/s',
                        'Flow Bytes/s', 'Fwd Packets/s', 'Bwd Header Length',
                        'Flow Packets/s', 'Avg Fwd Segment Size', 'Fwd Packet Length Mean',
                        'Bwd Packet Length Std', 'Flow IAT Mean','Flow IAT Std',
                        'Fwd Packet Length Std', 'Total Backward Packets',
                        'Subflow Bwd Packets', 'Bwd IAT Max', 'Subflow Fwd Packets',
                        'Total Fwd Packets', 'Bwd IAT Total', 'Bwd IAT Mean', 'Active Mean',
                        'Active Min', 'Active Max', 'Bwd Packet Length Min', 'Idle Max', 'Bwd IAT Std']

    X = data[columns_selected].values


    ipca = IncrementalPCA(n_components=n_features, batch_size=500)
    for batch in np.array_split(X, len(X) // 500):
        ipca.partial_fit(batch)

    transformed_features = ipca.transform(X)

    new_data = pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(n_features)])

    return new_data

def data_cleaning(data, n_featues = 22):
    """
        This function will clean incoming data. It leverages the `DataPreprocessing` and
        `FeatureEngineering` methods from the `src.utils` module to preprocess and
        feature engineer the data, respectively.

        Parameters:
        data (pandas.DataFrame): Input raw data which needs to be cleaned.

        Returns:
        pandas.DataFrame: Preprocessed and feature-engineered data.

        Usage:
        data = pd.read_csv(path)
        cleaned_data = data_cleaning(data)
    """
    data = preprocessing(data)
    data = feature_engineering(data, n_featues)
    return data



