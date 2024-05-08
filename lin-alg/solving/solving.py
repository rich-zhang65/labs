import numpy as np

a = np.array([[4,-3,1], [2,1,3], [-1,2,-5]], dtype = np.dtype(float))
b = np.array([-10,0,17], dtype = np.dtype(float))
print("Matrix A:")
print(a)
print("\nArray B:")
print(b)

print(f"Shape of A: {np.shape(a)}")
print(f"Shape of B: {np.shape(b)}")

x = np.linalg.solve(a, b)
print(f"Solution: {x}")

d = np.linalg.det(a)
print(f"Determinant of matrix A: {d:0.2f}")

a_2 = np.array([[1,1,1], [0,1,-3], [2,1,5]], dtype = np.dtype(float))
b_2 = np.array([2,1,0], dtype = np.dtype(float))

try:
    x_2 = np.linalg.solve(a_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)

d_2 = np.linalg.det(a_2)
print(f"Determinant of matrix A_2: {d_2:0.2f}")
