#A8
import pandas as pd
import numpy as np
df=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
df.replace('?', np.nan, inplace=True)
print("Missing values before adding:\n")
print(df.isnull().sum())
df_imputed=df.copy()

for col in df.columns:
    if df[col].dtype in ['int64','float64']:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lower=q1-1.5*iqr
        upper=q3+1.5*iqr
        has_outliers=((df[col]<lower)|(df[col]>upper)).any()

        if has_outliers:
            df_imputed[col].fillna(df[col].median(),inplace=True)
        else:
            df_imputed[col].fillna(df[col].mean(),inplace=True)

    else:
        df_imputed[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after adding:\n")
print(df_imputed.isnull().sum())
