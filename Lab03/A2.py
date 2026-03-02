#A2
# importing file
df = pd.read_excel("/content/BERT_Embeddings.xlsx", engine="openpyxl")


# mean and var and standard deviation own function
def meanByFormula(data):
  mean = sum(data)/len(data)
  return mean

def varByFormula(data):
  mean = meanByFormula(data)
  total = 0
  for x in data:
    total += (x - mean) ** 2
  var = total / len(data)
  return var

def standardDeviationByFormula(data):
  var = varByFormula(data)
  std = var ** 0.5
  return std

# producing mean, var and std dev for all features ie columns
def dataset_statistics_df(df):
    means = {}
    variances = {}
    stds = {}

    for col in df.columns:
        data = df[col].tolist()
        means[col] = meanByFormula(data)
        variances[col] = varByFormula(data)
        stds[col] = standardDeviationByFormula(data)

    return means, variances, stds


#mean for each class also known as class centroid
def class_centroid(df_class):
    return df_class.mean(axis=0).values

#spread for each class also known as standard deviation
def class_spread(df_class):
    return df_class.std(axis=0).values

# inter class or norm bw mean vectors of a specific class
def interclass_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

# dataset_statistics of are data
features = df.drop(columns=["label"])
features = features.select_dtypes(include=["int64", "float64"])
means, variances, stds = dataset_statistics_df(features)
print("mean of each features in data:",means)
print("variance of each feature in data:",variances)
print("standard deviation of each feature in data:",stds)

# separate features and class label and drps it
features = df.drop(columns=["label"])
features = features.select_dtypes(include=["int64", "float64"])
# takes two specific class
class1 = features[df["label"] == 1]
class2 = features[df["label"] == 2]

# centroids
c1 = class_centroid(class1)
c2 = class_centroid(class2)

# spreads
spread1 = class_spread(class1)
spread2 = class_spread(class2)

# distance
dist = interclass_distance(c1, c2)
