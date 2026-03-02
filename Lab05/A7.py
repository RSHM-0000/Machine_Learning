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
