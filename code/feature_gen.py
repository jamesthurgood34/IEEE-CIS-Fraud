from functools import reduce
import pandas as pd
from sklearn.model_selection import train_test_split


def OneHotEncode(df, cols):

    dfs = [pd.get_dummies(df[i], prefix=i) for i in cols]
    features = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
    # clean column names
    features.columns = [i.replace(' ', '_').replace('.', '_') for i in features.columns.to_list()]
    return features


def split_data(features, split):
    """
    Y_train, Y_test, X_train, X_test
    :param features:
    :return:
    """
    train_features, test_features = train_test_split(features, test_size=split)

    return (train_features.pop('isFraud'),
            test_features.pop('isFraud'),
            train_features.to_numpy(),
            test_features.to_numpy())
