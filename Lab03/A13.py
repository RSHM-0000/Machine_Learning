# performance metrics by own function
acc, prec, rec, f1 = confusion_metrics(y_test,y_pred)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
