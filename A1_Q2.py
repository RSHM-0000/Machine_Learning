# Q2

my_list=[] #We create an empty list

def get_range(my_list): #We find the range using a function
  return max(my_list) - min(my_list) #Given: Range=min - max

n=int(input("Enter number of elements: "))
if n<3:
  print("Range determination is not possible") #For checking if the list is small
else:
  for i in range(n): #We append the list till n elements
    ele = int(input(f"Enter element {i+1}: "))
    my_list.append(ele)

  print(get_range(my_list)) #We call the function to display results
