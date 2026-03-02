# fixing random values
np.random.seed(42)

# Generate 20 data points (values between 1 and 10)
X = np.random.randint(1, 11, size=20)
Y = np.random.randint(1, 11, size=20)

# Assign classes
# Class 0 (Blue): X + Y <= 10
# Class 1 (Red):  X + Y > 10
labels = np.where(X + Y > 10, 1, 0)

# Scatter plot
plt.figure()
plt.scatter(X[labels == 0], Y[labels == 0], color='blue', label='Class 0 (Blue)')
plt.scatter(X[labels == 1], Y[labels == 1], color='red', label='Class 1 (Red)')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Training Data Scatter Plot')
plt.legend()
plt.show()
