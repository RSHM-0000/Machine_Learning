import numpy as np
A=np.array([4,3,2,7,8,9])
B=np.array([33,6,2,7,0,2])

def dot_prod(a,b):
  print(np.dot(a,b))

def len_norm(a):
  print(np.linalg.norm(a))

def manual_dot_prod(a,b):
  print(sum(a[i]*b[i] for i in range(len(b))))

def manual_norm(a):
  print(np.sqrt(sum(a[i]**2 for i in range(len(a)))))

dot_prod(A,B)
len_norm(A)
len_norm(B)
manual_dot_prod(A,B)
manual_norm(A)
manual_norm(B)


