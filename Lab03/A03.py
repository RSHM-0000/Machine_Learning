# making buckets(data in range)
def histogram_data(num_bins,data):
  count,bin_edges = np.histogram(data,bins=num_bins)
  return count,bin_edges

# ploting a histogram with a feature
def plot_histogram(num_bins,data,feature_name="Feature"):
  count,bin_edges = histogram_data(num_bins,data)
  plt.hist(data,bins=num_bins,edgecolor='black')
  plt.xlabel(feature_name)
  plt.ylabel("Frequency")
  plt.title(f"Histogram of {feature_name}")
  plt.show()


num_bins = 5
data =   features.iloc[:, 0]
counts,bin_edges = histogram_data(num_bins,data)
print("Counts in each bucket:", counts)
print("Bucket edges:", bin_edges)
plot_histogram(3, data, feature_name=data.name)
