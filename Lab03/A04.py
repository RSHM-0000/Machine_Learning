#calculating minkowski distance
def minkowski_distance(A,B,p):
  sum=0
  for i in range(len(A)):
    sum = sum + abs(A[i]-B[i])**p
  return sum**(1/p)

# ploting graph of minkowski distance where p in range 1 to 10
def plot_graph(list,title):
  p_value = [i for i in range(1,len(list)+1)]
  plt.plot(p_value,list)
  plt.xlabel("p value")
  plt.ylabel("dist")
  plt.title(title)

A = features.iloc[0].values
B = features.iloc[1].values
minkowski_dis = [minkowski_distance(A,B,p) for p in range(1,11)]

print("Minkwoski Distance",minkowski_dis)
plot_graph(minkowski_dis,"Minkowski Distance")
