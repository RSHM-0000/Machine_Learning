def evaluate_clustering(X_scaled, labels):
    """Computes Silhouette, CH, and DB scores"""

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    return sil, ch, db

sil_score, ch_score, db_score = evaluate_clustering(X_scaled,kmeans.labels_)
print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_score)
