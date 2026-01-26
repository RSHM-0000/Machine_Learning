#A9
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
#Data normalization techniques are applied for those attributes where there is range for example price, age, volume
#Data normalization techniques are not applied for those attributes where there is binary datatypes for example the gender, 
#Possibility of goitre, pregnancy, etc as seen in thyroid dataset.

df.replace('?', np.nan, inplace=True)
num_cols=df.select_dtypes(include=['int64','float64']).columns
print("Numeric columns to normalize:", num_cols)

#Minmax- rescales data to a fixed range 0 to 1
minmax=MinMaxScaler()
df_minmax= df.copy()
df_minmax[num_cols]=minmax.fit_transform(df[num_cols])

print("\nMin-Max scaled data\n:")
print(df_minmax[num_cols].head())

#Z-Score Standardization- converts data to have mean=0 and standarad deviation=1
zscore=StandardScaler()
df_zscore=df.copy()
df_zscore[num_cols]=zscore.fit_transform(df[num_cols])

print("\nZ-Score standard data:")
print(df_zscore[num_cols].head())
