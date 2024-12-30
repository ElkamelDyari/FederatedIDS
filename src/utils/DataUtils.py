import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
import pandas as pd
from imblearn.over_sampling import SMOTE
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

    # Creating a dictionary that maps each label to its attack type
    attack_map = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'PortScan': 'Port Scan',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Bot',
        'Web Attack-Brute Force': 'Web Attack',
        'Web Attack-XSS': 'Web Attack',
        'Web Attack-Sql Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Heartbleed'
    }

    # Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
    data['Label'] = data['Label'].map(attack_map)

    le = LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])



    # dropping features with one unique value
    num_unique = data.nunique()
    not_one_variable = num_unique[num_unique > 1].index
    data = data[not_one_variable]

    return data


def feature_engineering(data):
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

    X = data.drop(['Label'], axis=1).values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    importances = mutual_info_classif(X_train, y_train)
    # calculate the sum of importance scores
    features = data.dtypes[data.dtypes != 'object'].index
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])

    # select the important features from top to bottom until the accumulated importance reaches 90%
    f_list2 = sorted(zip(map(lambda x: round(x, 4), importances / Sum), features), reverse=True)
    Sum2 = 0
    fs = []
    for i in range(0, len(f_list2)):
        Sum2 = Sum2 + f_list2[i][0]
        fs.append(f_list2[i][1])
        if Sum2 >= 0.9:
            break

    X_fs = data[fs].values

    ipca = IncrementalPCA()
    for batch in np.array_split(X_fs, len(X_fs) // 500):
        ipca.partial_fit(batch)

    # Explained variance ratio
    explained_variance_ratio = ipca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    size = np.argmax(cumulative_variance >= 0.99) + 1
    ipca = IncrementalPCA(n_components=size, batch_size=500)
    for batch in np.array_split(X_fs, len(X_fs) // 500):
        ipca.partial_fit(batch)

    transformed_features = ipca.transform(X_fs)

    new_data = pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(size)])
    new_data['Target'] = y

    return new_data

def data_cleaning(data):
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
    data = feature_engineering(data)
    return data


def over_sample(X_train, y_train):
    """
        This function applies Synthetic Minority Over-sampling Technique (SMOTE) on the provided training
        data in order to handle class imbalance for minority classes.
        It adds synthetic observations to the classes with labels 2, 4, and 5 until their count reaches 1000.

        Note that you need imbalanced-learn library installed to use this function (pip install -U imbalanced-learn)

        Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): The training data.
        y_train (numpy.ndarray or pandas.DataFrame): The corresponding training labels.

        Returns:
        tuple: The over-sampled training data and corresponding labels.
    """

    smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000, 4:1000, 5:1000})
    return smote.fit_resample(X_train, y_train)

def load_and_split_data(path, label = 'Target', test_size=0.2, random_state=0, smote=False):
    """
        This function performs the following steps:
        - Loads a dataset from a CSV file specified by the 'path' parameter.
        - Cleans this dataset by calling a predefined `data_cleaning` function on it.
        - Splits this cleaned dataset into features (X) and the target (y), according to the 'label' parameter.
        - Further splits these features and target into training and test sets.
        - If the 'smote' parameter is set to True, the Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training set,
          to handle class imbalance. Otherwise, data remains as it is.

        Parameters:
        path (str): The path to the CSV file which is to be loaded.
        label (str, optional): The label of the target variable to be predicted. Defaults to 'Target'.
        test_size (float, optional): The proportion of test size in the train-test split. Defaults to 0.2.
        random_state (int, optional): The seed for the random number generator during the split. Defaults to 0.
        smote (bool, optional): Whether SMOTE should be applied to the training dataset. Defaults to False.

        Returns:
        tuple: The training and test sets split from the input data.
               If 'smote' is True, these will be oversampled training sets.
               The tuple structure is: (X_train, X_test, y_train, y_test).
    """

    data = pd.read_csv(path)
    data = data_cleaning(data)
    X = data.drop([label], axis=1)
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if smote:
        X_train_smote, y_train_smote = over_sample(X_train, y_train)
        return X_train_smote, X_test, y_train_smote, y_test
    else:
        return X_train, X_test, y_train, y_test

def load_data(path):
    """
        This function performs the following steps:
        - Loads a dataset from a CSV file specified by the 'path' parameter.
        - Cleans this dataset by calling a predefined `data_cleaning` function on it.
        - Splits this cleaned dataset into features (X) and the target (y), by dropping the column 'Target'.
        - Further splits these features and target into training and test sets.

        Parameters:
        path (str): The path to the CSV file which is to be loaded.

        Returns:
        tuple: The training and test sets split from the input data. The tuple structure is: (X_train, X_test, y_train, y_test).
    """

    data = pd.read_csv(path)
    data = data_cleaning(data)
    X = data.drop(['Target'], axis=1)
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


