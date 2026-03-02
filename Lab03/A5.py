# comparing minkowski diatance with own fun and fun from scipy

d_own = minkowski_distance(A, B, p=2)  # Euclidean distance
print("Distance (own function):", d_own)

d_scipy = minkowski(A, B, p=2) # Euclidean distance
print("Distance (SciPy):", d_scipy)
