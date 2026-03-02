df = pd.read_excel('/content/Lab Session Data.xlsx', sheet_name=0)

X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y = df['Payment (Rs)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#  Metrics Function
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

#  Training Metrics
mse_train, rmse_train, mape_train, r2_train = regression_metrics(
    y_train, y_train_pred
)

#Testing Metrics
mse_test, rmse_test, mape_test, r2_test = regression_metrics(
    y_test, y_test_pred
)


print("TRAINING METRICS")
print("MSE  :", mse_train)
print("RMSE :", rmse_train)
print("MAPE :", mape_train)
print("R2   :", r2_train)

print("\nTESTING METRICS")
print("MSE  :", mse_test)
print("RMSE :", rmse_test)
print("MAPE :", mape_test)
print("R2   :", r2_test)
