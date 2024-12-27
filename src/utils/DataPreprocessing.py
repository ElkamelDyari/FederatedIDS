import numpy as np
from sklearn.preprocessing import LabelEncoder

def drop_duplicates(data):
    data.drop_duplicates(inplace=True)
    return data

def deal_with_missing_values(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['Flow Bytes/s'].fillna(data['Flow Bytes/s'].median(), inplace=True)
    data['Flow Packets/s'].fillna(data['Flow Packets/s'].median(), inplace=True)
    return data

def z_score_normalization(data):
    features = data.dtypes[data.dtypes != 'object'].index
    data[features] = data[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    return data

def create_target(data):
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
    return data

def preprocessing(data):

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
