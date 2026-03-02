import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Train-test split (7:3)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# kNN model
knn = KNeighborsClassifier()

# Hyperparameter grid
param_grid = {
    'n_neighbors': list(range(1, 21))
}

# Grid Search
grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# Best results
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print("Best k value:", best_k)
print("Best Cross-Validation Accuracy:", best_score)


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_neighbors': np.arange(1, 51)
}

random_search = RandomizedSearchCV(
    knn,
    param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best k value:", random_search.best_params_['n_neighbors'])
print("Best Cross-Validation Accuracy:", random_search.best_score_)
