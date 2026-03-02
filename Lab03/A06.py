# here X are the features with which we need to Y which is which class it belong s to ie label
X = features.values
y = df["label"].values
# separates the training and the testing part in the ratio of 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
