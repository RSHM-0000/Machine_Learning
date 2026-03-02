#A6
#A3 TO A5 for our project data
df = pd.read_excel("/content/Stereotype ml dataset (1).xlsx", engine="openpyxl")
features = df.drop(columns=["Label"])
features = features.select_dtypes(include=["int64", "float64"])
X = features.values
y = df["Label"].values
# separates the training and the testing part in the ratio of 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#trains with the dataset got from the split with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
#class boundry for our project
y_pred1 = model.predict(X_test)
plot_knn(X_train, y_train, X_test, y_pred1, 3)
