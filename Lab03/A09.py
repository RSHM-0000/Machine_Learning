#prediction for the test dataset
y_pred = neigh.predict(X_test)
print(y_pred)
#predicts for 1 feature in test dataset
test_vect = X_test[0]
print("Predicted class:", neigh.predict([test_vect]))
