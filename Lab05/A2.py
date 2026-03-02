from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np

# Reusing previously defined functions:
# load_data()
# prepare_data()
# train_model()
# predict()

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


def main():
    file_path = "Stereotype ml dataset (1).xlsx"
    
    df = load_data(file_path)          # reused
    X, y = prepare_data(df)            # reused
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train)  # reused
    
    y_train_pred = predict(model, X_train) # reused
    y_test_pred = predict(model, X_test)   # reused
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    print("TRAIN METRICS")
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
