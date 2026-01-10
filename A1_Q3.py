# Q3

import numpy as np #Importing numpy

def matrix_power(m): #We create a a fucntion for modular code
  a=int(input("Enter number of rows: "))
  b=int(input("Enter number of columns: "))
  if a!=b: #Checking if its square matrix
    print("Not square matrix")
  else:
    matrix=[] #We insert lists of rows in the matrix list to form 2d matrix
    for i in range (a):
      row=[]
      for j in range(b):
        row.append(int(input("Enter elements:"))) #takes elements until the column number
      matrix.append(row)
    print("The matrix is: ")
    for i in range(a):
      for j in range(b):
        print(matrix[i][j], end=" ") #Printing matrix
      print("\n")

    result = np.linalg.matrix_power(matrix,m) #Using numpy, matrix is raised to the power of m
    return(result)#Return reslut for the function

m=int(input("Enter the power of matrix: "))
print(matrix_power(m))#Print results
