import numpy as np
import matplotlib.pyplot as plt
import utils

a = np.array([[2, 3], [2, 1]])
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

utils.plot_transformation(a, e1, e2, vector_name = 'e')

a_eig = np.linalg.eig(a)
print("\n")
print(f"Matrix A:\n{a} \n\nEigenvalues of matrix A:\n{a_eig[0]}\n\nEigenvectors of matrix A:\n{a_eig[1]}")

utils.plot_transformation(a, a_eig[1][:,0].reshape(-1, 1), a_eig[1][:,1].reshape(-1, 1))

a_reflection_yaxis = np.array([[-1, 0], [0, 1]])
a_reflection_yaxis_eig = np.linalg.eig(a_reflection_yaxis)
print(f"Matrix A:\n {a_reflection_yaxis} \n\nEigenvalues of matrix A:\n {a_reflection_yaxis_eig[0]}", f"\n\nEigenvectors of matrix A:\n {a_reflection_yaxis_eig[1]}")

utils.plot_transformation(a_reflection_yaxis, a_reflection_yaxis_eig[1][:,0].reshape(-1, 1), a_reflection_yaxis_eig[1][:,1].reshape(-1, 1))

a_shear_x = np.array([[1, 0.5], [0, 1]])
a_shear_x_eig = np.linalg.eig(a_shear_x)
print(f"Matrix a_shear_x:\n {a_shear_x} \n\nEigenvalues of matrix a_shear_x:\n {a_shear_x_eig[0]}", f"\n\nEigenvectors of matrix a_shear_x \n {a_shear_x_eig[1]}")

utils.plot_transformation(a_shear_x, a_shear_x_eig[1][:,0].reshape(-1, 1), a_shear_x_eig[1][:,1].reshape(-1, 1))

a_rotation = np.array([[0, 1], [-1, 0]])
a_rotation_eig = np.linalg.eig(a_rotation)

print(f"Matrix a_rotation:\n {a_rotation}\n\nEigenvalues of matrix a_rotation:\n {a_rotation_eig[0]}", f"\n\nEigenvectors of matrix a_rotation \n {a_rotation_eig[1]}")

a_identity = np.array([[1, 0], [0, 1]])
a_identity_eig = np.linalg.eig(a_identity)

utils.plot_transformation(a_identity, a_identity_eig[1][:,0].reshape(-1, 1), a_identity_eig[1][:,1].reshape(-1, 1))

print(f"Matrix a_identity:\n {a_identity}\n\nEigenvalues of matrix a_identity:\n {a_identity_eig[0]}", f"\n\nEigenvectors of matrix a_identity\n {a_identity_eig[1]}")

a_scaling = np.array([[2, 0], [0, 2]])
a_scaling_eig = np.linalg.eig(a_scaling)

utils.plot_transformation(a_scaling, a_scaling_eig[1][:,0].reshape(-1, 1), a_scaling_eig[1][:,1].reshape(-1, 1))

print(f"Matrix a_scaling:\n {a_scaling}\n\nEigenvalues of matrix a_scaling:\n {a_scaling_eig[0]}", f"\n\nEigenvectors of matrix a_scaling\n {a_scaling_eig[1]}")

a_projection = np.array([[1, 0], [0, 0]])
a_projection_eig = np.linalg.eig(a_projection)

utils.plot_transformation(a_projection, a_projection_eig[1][:,0].reshape(-1, 1), a_projection_eig[1][:,1].reshape(-1, 1))

print(f"Matrix a_projection:\n {a_projection}\n\nEigenvalues of matrix a_projection:\n {a_projection_eig[0]}", f"\n\nEigenvectors of matrix a_projection\n {a_projection_eig[1]}")
