import numpy as np
df = pd.read_excel('Lab Session Data.xlsx')

X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
Y=df['Payment (Rs)'].values
print(f"The X matrix is \n{X}")
print(f"The Y matrix is {Y}")
dimensions=X.shape[1] #Shape[1] means the 2 dimension that is the column. Thus it gives the count of no of columns/features
print(f"The dimension of this data is {dimensions}")
X_no_vectors=X.shape[1]
print(f"The number vectors in X's vector space is {X_no_vectors}")
A=X*Y
rank_A=np.linalg.matrix_rank(A)
print(f"The rank of this matrix is {rank_A}")
rank=np.linalg.matrix_rank(X)
print(f"The rank of this matrix is {rank}")
