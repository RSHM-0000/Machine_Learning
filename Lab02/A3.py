import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
price=df['Price'].values
mean_np=np.mean(price)
var_np=np.var(price)
print(f"Mean from numpy:{mean_np}")
print(f"Variance from numpy: {var_np}")

def my_mean(arr):
    total=0
    for x in arr:
        total+=x
    return total/len(arr)
def my_var(arr):
    mu=my_mean(arr)
    total=0
    for x in arr:
        total+=(x-mu)**2
    return total/len(arr)

print("Mean manual:",my_mean(price))
print("Variance manual:", my_var(price))

def time_function(func, data, runs=10):
    times=[]
    for _ in range(runs):
        start= time.time()
        func(data)
        end=time.time()
        times.append(end-start)
    return sum(times)/runs
numpy_mean_time=time_function(np.mean, price)
custom_mean_time=time_function(my_mean, price)

numpy_var_time=time_function(np.var,price)
custom_var_time=time_function(my_var,price)

print(f"NumPy Mean Time: {numpy_mean_time}")
print(f"Custom Mean Time: {custom_mean_time}")

print(f"NumPy Variance Time: {numpy_var_time}")
print(f"Custom Variance Time: {custom_var_time}")

wednesday_prices=df[df['Day']=='Wednesday']['Price']

wed_price=df[df['Day']=='Wednesday']['Price']
wed_mean=wed_price.mean()

print(f"Wednesday Mean: {wed_mean}")
print(f"Population Mean: {mean_np}")


df['Date']=pd.to_datetime(df['Date'])

april_prices=df[df['Date'].dt.month==4]['Price']
april_mean=april_prices.mean()

print(f"April Mean: {april_mean}")
print(f"Population Mean: {mean_np}")

chg = df['Chg%'].values

loss_count=len(list(filter(lambda x:x<0,chg)))
prob_loss=loss_count/len(chg)

print(f"Probability of loss: {prob_loss}")
wed_chg=df[df['Day']=='Wednesday']['Chg%']

profit_wed=np.sum(wed_chg>0)
prob_profit_wed=profit_wed/len(wed_chg)

print(f"Probability of profit on Wednesday: {prob_profit_wed}")
print(f"Conditional Probability of Profit given Wednesday: {prob_profit_wed}")
plt.figure()
plt.scatter(df['Day'], df['Chg%'])
plt.xlabel("Day of week")
plt.ylabel("Chg%")
plt.title("Chg% vs Day of week")
plt.show()
