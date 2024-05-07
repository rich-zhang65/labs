import numpy as np

two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)

one_dim_arr = np.array([1, 2, 3, 4, 5, 6])
multi_dim_arr = np.reshape(one_dim_arr, (2, 3))
print(multi_dim_arr)

print(multi_dim_arr.ndim)
print(multi_dim_arr.shape)
print(multi_dim_arr.size)
