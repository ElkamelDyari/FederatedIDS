from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pandas as pd


def feature_engineering(data):
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
