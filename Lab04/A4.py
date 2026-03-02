import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Data Functions
def generate_training_data(n=20):
    np.random.seed(42)
    x1 = np.random.randint(1, 11, n)
    x2 = np.random.randint(1, 11, n)
    y = np.where(x1 + x2 > 10, 1, 0)
    return np.column_stack((x1, x2)), y

def generate_test_data():
    x = np.arange(0, 10.1, 0.1)
    y = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x, y)
    return np.c_[xx.ravel(), yy.ravel()]

# kNN Function
def knn_predict(X_train, y_train, X_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Plot Function
def plot_knn(X_train, y_train, X_test, y_pred, k):
    plt.figure()
    plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], s=5, label='Class 0 (Blue)')
    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], s=5, label='Class 1 (Red)')
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                edgecolors='black', marker='o', label='Train Class 0')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                edgecolors='black', marker='^', label='Train Class 1')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(f"kNN Classification (k = {k})")
    plt.legend(markerscale=2)
    plt.show()


X_train, y_train = generate_training_data()
X_test = generate_test_data()

# A4: k = 3
y_pred = knn_predict(X_train, y_train, X_test, k=3)
plot_knn(X_train, y_train, X_test, y_pred, k=3)
