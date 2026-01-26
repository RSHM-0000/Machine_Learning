#A5
df=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

doc1=df.iloc[0]
doc2=df.iloc[1]
binary_cols=[]

for col in df.columns:
    unique_vals=set(df[col].dropna().unique())
    if unique_vals.issubset({0,1,'0','1','t','f','Yes','No'}):
        binary_cols.append(col)

def to_binary(x):
    if x in [1,'1',True,'t']:
        return 1
    return 0
doc1_bin=doc1[binary_cols].apply(to_binary)
doc2_bin=doc2[binary_cols].apply(to_binary)

f11=((doc1_bin==1)&(doc2_bin==1)).sum()
f10=((doc1_bin==1)&(doc2_bin==0)).sum()
f01=((doc1_bin==0)&(doc2_bin==1)).sum()
f00=((doc1_bin==0)&(doc2_bin==0)).sum()

jc= f11/(f11+f10+f01)
smc =(f11+f00)/(f11+f10+f01+f00)

# Output
print(f"Binary attributes used: {binary_cols}")
print("f11:",f11,"f10:",f10,"f01:",f01,"f00:",f00)
print(f"JC: {jc}")
print(f"SMC:{smc}")
