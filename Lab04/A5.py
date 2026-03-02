#Repeat for different k values
for k in [1, 5, 7]:
    y_pred = knn_predict(X_train, y_train, X_test, k)
    plot_knn(X_train, y_train, X_test, y_pred, k)
