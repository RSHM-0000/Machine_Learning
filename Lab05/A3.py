from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np

# Reused from previous version:
# load_data()
# split_data()
# train_model()
# predict()
# evaluate_model()

def prepare_data_multi(df):
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, 0].values
    return X, y


def main():
    file_path = "Stereotype ml dataset (1).xlsx"
    
    df = load_data(file_path)                 # reused
    X, y = prepare_data_multi(df)             # new
    
    X_train, X_test, y_train, y_test = split_data(X, y)  # reused
    
    model = train_model(X_train, y_train)     # reused
    
    y_train_pred = predict(model, X_train)    # reused
    y_test_pred = predict(model, X_test)      # reused
    
    train_metrics = evaluate_model(y_train, y_train_pred)  # reused
    test_metrics = evaluate_model(y_test, y_test_pred)     # reused
    
    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    
    print("\nTRAIN METRICS")
    print("MSE:", train_metrics[0])
    print("RMSE:", train_metrics[1])
    print("MAPE:", train_metrics[2])
    print("R2:", train_metrics[3])
    
    print("\nTEST METRICS")
    print("MSE:", test_metrics[0])
    print("RMSE:", test_metrics[1])
    print("MAPE:", test_metrics[2])
    print("R2:", test_metrics[3])


if __name__ == "__main__":
    main()
