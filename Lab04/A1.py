# separates the training and the testing part in the ratio of 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#trains with the dataset got from the split with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

#confusion matrix precision recall f1score for the test dataset
def performance_metrics(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    return cm, precision, recall, f1

cm_train, p_train, r_train, f1_train = performance_metrics(model, X_train, y_train)
cm_test, p_test, r_test, f1_test = performance_metrics(model, X_test, y_test)

# infering output
def infer_learning_outcome(f1_train, f1_test):
    if f1_train > 0.9 and f1_test < 0.7:
        return "Overfitting"
    elif f1_train < 0.7 and f1_test < 0.7:
        return "Underfitting"
    else:
        return "Regular fitting (Good Generalization)"

print("Training Confusion Matrix:\n", cm_train)
print("Training -> Precision:", p_train, "Recall:", r_train, "F1:", f1_train)

print("\nTest Confusion Matrix:\n", cm_test)
print("Test -> Precision:", p_test, "Recall:", r_test, "F1:", f1_test)
outcome = infer_learning_outcome(f1_train, f1_test)
print("\nModel Learning Outcome:", outcome)

