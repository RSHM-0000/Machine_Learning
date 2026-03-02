def plot_scores(k_values, sil_scores, ch_scores, db_scores):
    # Silhouette Score
    plt.figure()
    plt.plot(k_values, sil_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.grid(True)
    plt.show()

    # Calinski–Harabasz Score
    plt.figure()
    plt.plot(k_values, ch_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Calinski-Harabasz Score")
    plt.title("Calinski-Harabasz Score vs k")
    plt.grid(True)
    plt.show()

    # Davies–Bouldin Index
    plt.figure()
    plt.plot(k_values, db_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Davies-Bouldin Index vs k")
    plt.grid(True)
    plt.show()


# Scale features
X_scaled = scale_features(X)

# Range of k values
k_values = range(2, 11)

silhouette_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = apply_kmeans(X_scaled, n_clusters=k)
    sil, ch, db = evaluate_clustering(X_scaled, kmeans.labels_)
    
    silhouette_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)

plot_scores(k_values, silhouette_scores, ch_scores, db_scores)
