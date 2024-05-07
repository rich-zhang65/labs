import numpy as np

test = "Hello World"
print(test)

one_dimensional_arr = np.array([10, 12])
print(one_dimensional_arr)

a = np.array([1, 2, 3])
print(a)

b = np.arange(3)
print(b)

c = np.arange(1, 20, 3)
print(c)

lin_spaced_arr = np.linspace(0, 100, 5)
print(lin_spaced_arr)

lin_spaced_arr_int = np.linspace(0, 100, 5, dtype=int)
print(lin_spaced_arr_int)

c_int = np.arange(1, 20, 3, dtype=int)
print(c_int)

b_float = np.arange(3, dtype=float)
print(b_float)

char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr)
print(char_arr.dtype)

ones_arr = np.ones(3)
print(ones_arr)

zeros_arr = np.zeros(3)
print(zeros_arr)

empt_arr = np.empty(10)
print(empt_arr)

rand_arr = np.random.rand(3)
print(rand_arr)
