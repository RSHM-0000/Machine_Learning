my_list=[2,7,4,1,3,6]
count=0 #Variable to get the count of pairs
n=len(my_list) #No of elements

for i in range(n): #Traverse the list using index instead of the items
  for j in range(i+1,n): #Traverse the list after the index i, to prevent extra count of redundant pairs like (7,3) & (3,7)
    if my_list[i]+my_list[j]==10: #Check for the condition
      count+=1 #Increase the count

print(f"The number of pairs that add up to 10= {count}")
