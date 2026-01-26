#A6
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def cosine_similarity(v1,v2):
    dot_prod=np.dot(v1,v2)
    vec1=np.linalg.norm(v1)
    vec2=np.linalg.norm(v2)
    if vec1==0 or vec2==0:
        return 0
    return dot_prod/(vec1*vec2)


df_encoded=df.copy()
le=LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype=='object':
        df_encoded[col]=le.fit_transform(df_encoded[col].astype(str))

A=df_encoded.iloc[0].values
B=df_encoded.iloc[1].values
similarity=cosine_similarity(A,B)

print(f"Cosine similarity: {similarity}")
