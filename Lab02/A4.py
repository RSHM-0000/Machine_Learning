# A4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
df.head()
df.info()
#If float or int then the data is numerical otherwise it is categorical
#Nominal -no order, ordinal - order, numerical -0-9 no.s and binary -0,1
#And for nominal variables-One-hot encoding & for ordinal variables-label enocding, binary same label encoding
#here sex is either female or male which means nominal variable which is One-hot encoding. Also referral solution is nominal too
#Age is numerical
#Pregnant, goitre, tumor, hypopituitary in binary

print("Data types:\n")
for col in df.columns:
    if df[col].dtype=='object':
        print(f"{col}:Categorical(Nominal or binary)")
    else:
        print(f"{col}:Numeric(Continuous)")

print("\nEncoding\n")
for col in df.columns:
    if df[col].dtype =='object':
        unique_vals=df[col].nunique()
        if unique_vals ==2:
            print(f"{col}:Label Encoding(binary)")
        else:
            print(f"{col}:One-Hot Encoding(nominal)")

print("\nDatatypes type for numerical datatypes")
numeric_cols=df.select_dtypes(include=['int64', 'float64'])
print(numeric_cols.describe())

print("\nmissing values with '?'")
print((df=='?').sum())

print("\nOutliers in data\n")
for col in numeric_cols.columns:
    q1=df[col].quantile(0.25)
    q3 =df[col].quantile(0.75)
    iqr =q3-q1
    outliers=df[(df[col]< q1-1.5*iqr) |(df[col]>q3+1.5*iqr)]
    print(f"{col}:Number of outliers= {len(outliers)}")


mean =numeric_cols.mean()
variance =numeric_cols.var()

print(f"Means:\n {mean}")
print(f"\nVariances:\n {variance}")
