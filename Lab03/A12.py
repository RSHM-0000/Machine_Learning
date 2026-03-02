def evaluate_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return cm, acc, prec, rec, f1
cm_test, acc_test, prec_test, rec_test, f1_test = evaluate_classification(y_test, y_pred_test)

print("Test Confusion Matrix:\n", cm_test)
print("Test Accuracy:", acc_test)
print("Test Precision:", prec_test)
print("Test Recall:", rec_test)
print("Test F1 Score:", f1_test)
