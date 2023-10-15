import numpy as np

list_a = [1, 2, 3]
# arr_ = [np.array([i, i**2]).flatten() for i in arr]
list_b = []
[list_b.extend([i, i**2]) for i in list_a]
# print(arr_)
print(list_b)