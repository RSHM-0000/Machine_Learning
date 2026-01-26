#A2

from sklearn.linear_model import LogisticRegression
df=pd.read_excel('Lab Session Data.xlsx')
df['Class']=np.where(df['Payment (Rs)']>200,'RICH','POOR')
df['Classlabel']=np.where(df['Class']=='RICH',1,0)
x=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
y=df['Classlabel'].values
model=LogisticRegression()
model.fit(x,y)
predictions = model.predict(x)
print(predictions)
