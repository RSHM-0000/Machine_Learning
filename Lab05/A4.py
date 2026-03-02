X = features.values
#or X = df.drop("Label")
y = df["Label"].values  # target (NOT used in clustering)
#A4

#Feature scaling (VERY IMPORTANT for K-means)
def scale_features(X):
    """
    Scales the feature matrix using StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


#Apply K-means clustering

def apply_kmeans(X_scaled, n_clusters=2, random_state=0):
    """
    Applies K-means clustering
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto"
    )
    kmeans.fit(X_scaled)
    return kmeans


# Feature scaling
X_scaled = scale_features(X)

# Apply K-means clustering
kmeans = apply_kmeans(X_scaled, n_clusters=2, random_state=0)

# Cluster labels
cluster_labels = kmeans.labels_
print("Cluster labels:")
print(cluster_labels)

# Cluster centers
cluster_centers = kmeans.cluster_centers_
print("Cluster centers:")
print(cluster_centers)


#A5


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



#A6

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


#A7

def elbow_method(X_scaled, k_range=range(2, 20)):
    """Computes distortions (inertia) for different k values"""
    distortions = []

    for k in k_range:
        kmeans = apply_kmeans(X_scaled, n_clusters=2, random_state=0)
        distortions.append(kmeans.inertia_)

    return distortions

def plot_elbow(k_values, distortions):
    plt.figure()
    plt.plot(k_values, distortions, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()


k_values = range(2, 20)
distortions = elbow_method(X_scaled, k_values)
plot_elbow(k_values, distortions)
