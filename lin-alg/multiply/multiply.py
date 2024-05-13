import numpy as np

a = np.array([[4,9,9], [9,1,6], [9,2,3]])
print("Matrix A (3 x 3):\n", a)

b = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 x 2):\n", b)

print(np.matmul(a, b))
print(a @ b)

try:
    np.matmul(b, a)
except ValueError as err:
    print(err)

try:
    b @ a
except ValueError as err:
    print(err)

x = np.array([1, -2, -5])
y = np.array([4, 3, -1])
print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

print(np.matmul(x, y))

try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)

print(np.dot(a, b))
print(a - 2)
