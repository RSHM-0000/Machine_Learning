# creating our own knn classifiers
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], x_test)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [label for _, label in neighbors]
    return max(set(labels), key=labels.count)

    # creating our own confusion matrix
def confusion_metrics(y_true, y_pred):
    TP = FP = FN = TN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: TP += 1
        elif t == 0 and p == 1: FP += 1
        elif t == 1 and p == 0: FN += 1
        else: TN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, precision, recall, f1


own_preds = [knn_predict(X_train, y_train, x, 3) for x in X_test]
print("Own kNN accuracy:", confusion_metrics(y_test, own_preds)[0])
