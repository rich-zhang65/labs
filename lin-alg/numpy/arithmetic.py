import numpy as np

arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

addition = arr_1 + arr_2
print(addition)

subtraction = arr_1 - arr_2
print(subtraction)

# Only multiplies same index elements together, returns same sized array as both operands
multiplication = arr_1 * arr_2
print(multiplication)

vector = np.array([1, 2])
vector = vector * 1.6
print(vector)
