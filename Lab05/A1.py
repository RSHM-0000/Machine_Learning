import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score 
from sklearn.metrics import davies_bouldin_score
#optional scaling 
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(file_path):
    """
    Load Excel file into a pandas DataFrame
    """
    df = pd.read_excel(file_path)
    return df


def prepare_data(df):
    """
    Extract first column as X and second column as y
    """
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    return X, y


def train_model(X, y):
    """
    Train Linear Regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict(model, X):
    """
    Make predictions
    """
    return model.predict(X)


def main():
    file_path = "Stereotype ml dataset (1).xlsx"
    
    df = load_data(file_path)
    X, y = prepare_data(df)
    
    model = train_model(X, y)
    y_pred = predict(model, X)
    
    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)


if __name__ == "__main__":
    main()
