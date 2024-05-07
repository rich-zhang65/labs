import numpy as np
import matplotlib.pyplot as plt
from utils import plot_lines

a = np.array([[-1,3], [3,2]], dtype = np.dtype(float))
b = np.array([7,1], dtype = np.dtype(float))
print("Matrix A:")
print(a)
print("\nArray B:")
print(b)

print(f"Shape of A: {a.shape}")
print(f"Shape of B: {b.shape}")

x = np.linalg.solve(a, b)
print(f"Solution: {x}")

d = np.linalg.det(a)
print(f"Determinant of matrix A: {d:0.2f}")

a_system = np.hstack((a, b.reshape((2, 1))))
print(a_system)
print(a_system[1])

plot_lines(a_system)

a_2 = np.array([[-1,3], [3,-9]], dtype = np.dtype(float))
b_2 = np.array([7,1], dtype = np.dtype(float))
d_2 = np.linalg.det(a_2)
print(f"Determinant of matrix A_2: {d_2:0.2f}")

try:
    x_2 = np.linalg.solve(a_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)

a_2_system = np.hstack((a_2, b_2.reshape((2, 1))))
print(a_2_system)

plot_lines(a_2_system)

b_3 = np.array([7,-21], dtype = np.dtype(float))
a_3_system = np.hstack((a_2, b_3.reshape((2, 1))))
print(a_3_system)

plot_lines(a_3_system)
