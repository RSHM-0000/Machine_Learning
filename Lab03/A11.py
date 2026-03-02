#accuracy plot for k=1 to 12
k_vals = range(1, 12)
accs = []
for k in k_vals:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    accs.append(clf.score(X_test, y_test))

plt.plot(k_vals, accs, marker='o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.show()
