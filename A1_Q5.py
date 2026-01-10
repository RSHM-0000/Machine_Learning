#Q5

import random, statistics #Can generate random numbers using random module
#Statistics module can be used for doing mean, median and mode using functions

my_list=[]
for i in range(0,25):#Generate for 25 numbers
  number=random.randint(1,10) #Taking random integers between 1 to 10
  my_list.append(number)

print(my_list)
print("The mean of the numbers generated is: ",statistics.mean(my_list))#Computing mean
print("The mean of the numbers generated is: ",statistics.median(my_list))#Computing median
print("The mean of the numbers generated is: ",statistics.mode(my_list))#Computing mode
