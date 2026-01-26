import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
df_20=df.iloc[:20]
df_encoded=df_20.copy()
le=LabelEncoder()

for col in df_encoded.columns:
    if df_encoded[col].dtype=='object':
        df_encoded[col]= le.fit_transform(df_encoded[col].astype(str))

binary_cols=[]
for col in df_encoded.columns:
    unique_val= set(df_encoded[col].unique())
    if unique_val.issubset({0, 1}):
        binary_cols.append(col)

binary_data= df_encoded[binary_cols].values
full_data= df_encoded.values

def jaccard(v1,v2):
    f11=np.sum((v1==1)&(v2==1))
    f10=np.sum((v1==1)&(v2==0))
    f01=np.sum((v1==0)&(v2==1))
    return f11/(f11+f10+f01) if (f11+f10+f01)!=0 else 0


def smc(v1,v2):
    f11= np.sum((v1==1)&(v2==1))
    f10=np.sum((v1==1)&(v2==0))
    f01=np.sum((v1==0)&(v2==1))
    f00=np.sum((v1==0)&(v2==0))
    return (f11+f00)/(f11+f10+f01+f00)

def cosine(v1,v2):
    norm_v1=np.linalg.norm(v1)
    norm_v2=np.linalg.norm(v2)
    if norm_v1==0 or norm_v2==0:
        return 0.0
    return np.dot(v1,v2)/(norm_v1*norm_v2)


n=20
jc_matrix=np.zeros((n,n))
smc_matrix=np.zeros((n,n))
cos_matrix=np.zeros((n,n))

for i in range(n):
    for j in range(n):
        jc_matrix[i,j]= jaccard(binary_data[i],binary_data[j])
        smc_matrix[i,j]= smc(binary_data[i],binary_data[j])
        cos_matrix[i,j]=cosine(full_data[i],full_data[j])

plt.figure(figsize=(7,6))
sns.heatmap(jc_matrix,annot=True, cmap="viridis")
plt.title("Jaccard coeffecient heatmap")
plt.show()

plt.figure(figsize=(7,6))
sns.heatmap(smc_matrix,annot=True, cmap="viridis")
plt.title("Smc heatmap")
plt.show()

plt.figure(figsize=(7,6))
sns.heatmap(cos_matrix,annot=True, cmap="viridis")
plt.title("Cos heatmap")
plt.show()
