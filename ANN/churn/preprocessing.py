import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def process_churn():
    dataset = pd.read_csv('../datasets/Churn_Modelling.csv')
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])
    transformers = [('encoder', OneHotEncoder(), [1])]
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
