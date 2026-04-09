import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

def load_data(file_path):
    data = np.load(file_path)

    print("Available keys:", data.files)

    X = data['features']
    y = data['labels']

    print("X Shape:", X.shape)
    print("y Shape:", y.shape)

    return X, y

def preprocess_data(X, y):
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def apply_pca(X_train, X_test, n_components=200):
    pca = PCA(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Explained variance ratio sum:", sum(pca.explained_variance_ratio_))

    return X_train_pca, X_test_pca

def tune_model(X_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt']
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("\nBest Parameters:", random_search.best_params_)
    print("Best CV Score:", random_search.best_score_)

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, zero_division=0))

def run_pipeline(file_path):
    X, y = load_data(file_path)

    X, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = apply_pca(X_train, X_test, n_components=200)

    model = tune_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

run_pipeline("fc7_features.npz")
