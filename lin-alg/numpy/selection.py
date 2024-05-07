import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a[2])
print(a[0])

two_dim = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
print(two_dim[2][1])
print(two_dim[2,1])

# Excludes end (idx 4)
sliced_arr = a[1:4]
print(sliced_arr)

sliced_arr = a[:3]
print(sliced_arr)

sliced_arr = a[2:]
print(sliced_arr)

# [start:end:step] -> using [::step] will always return first element no matter step
sliced_arr = a[::2]
print(sliced_arr)

print(np.logical_and(a == a[:], a[:] == a[::]))

sliced_arr_1 = two_dim[0:2]
print(sliced_arr_1)

sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)

# Good for grabbing all elements at a certain column in all arrays
sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)
