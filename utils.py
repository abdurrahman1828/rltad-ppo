# utils

import numpy as np
import pandas as pd

def dataframe_to_matrix(df, startcol=0):
    cols = df.shape[1] - 1
    x = np.zeros(shape=(df.shape[0], cols))
    for i in range(cols):
        x[:, i] = df.iloc[:, i + startcol]
    labels = df['anomaly'] # adjust according to dataset
    return x, labels


def read_data_as_matrix(data):
    """ Reads data from CSV file and returns numpy matrix.
    Important: Assumes that the last column has the label \in {anomaly:1, nominal:0}
    :param datapath: the path of the data file
    :return: numpy.ndarray
    """
    X_train, labels = dataframe_to_matrix(data)
    anomalies = np.array(labels.index[labels == 1])
    return X_train, labels, anomalies


def labeling(x):
    l = len(x)
    new = []
    for i in range(l):
        if x[i] == 1:
            new.append(0)
        else:
            new.append(1)
    return new


def run_iforest(X):
    """ Predict the anomaly score with iForest; The lower, the more abnormal.
    Negative scores represent outliers, positive scores represent inliers.
    """
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest()
    clf.fit(X)
    scores = clf.predict(X)
    scores = labeling(scores)
    return scores


def data_split(X, testset_ratio=0.5, period=1):
    """ Splits the whole time series in train and test sets.
    :param X: data,
    :param testset_ratio: fraction of data used for test set,
    :param period: periodicity. e.g. for day wise data period could be 24 hours,
    :return: trainset, testset
    """
    rows, columns = X.shape
    count_test = int((rows / period) * testset_ratio) * period
    count_train = rows - count_test
    dataset_train = X[:count_train]
    dataset_test = X[count_train:]
    return dataset_train, dataset_test


def scaling(X, columns_scaling):
    """ Scale the data with MinMaxScaler.
    :param X: data,
    :param columns_scaling: list of columns to scale,
    :return: Scaled data
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = StandardScaler()
    X[columns_scaling] = scaler.fit_transform(X[columns_scaling])
    return X


def sliding_wd(X, size=10, initial_index=0):
    col = X.columns[:-1]
    length = len(X[col[0]]) - size + 1
    feature = []
    label = []
    features = (np.asarray(X[col]))  # .tolist()
    labels = X[X.columns[-1]]
    ini = initial_index
    end = initial_index + size
    for i in range(length):
        f = features[ini:end]
        feature.append(f)
        l = labels[end - 1:end]
        l = np.array(l)
        label.append(l)
        ini += 1
        end += 1
        if end > len(X[col[0]]):
            break

    return feature, label


def flatten(t):  # makes a single list from list of lists
    return [item for sublist in t for item in sublist]
