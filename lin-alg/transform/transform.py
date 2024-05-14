import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils

v = np.array([[3], [5]])
w = utils.T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

u = np.array([[1], [-2]])
v = np.array([[2], [4]])

k = 7

print("T(k*v):\n", utils.T(k*v), "\n k*T(v):\n", k*utils.T(v), "\n\n")
print("T(u+v):\n", utils.T(u+v), "\n T(u)+T(v):\n", utils.T(u)+utils.T(v))

v = np.array([[3], [5]])
w = utils.L(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = utils.transform_vectors(utils.T_hscaling, e1, e2)
print("Original vectors:\n e1 = \n", e1, "\n e2=\n", e2, "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

utils.plot_transformation(utils.T_hscaling, e1, e2)

e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_reflection_yaxis = utils.transform_vectors(utils.T_reflection_yaxis, e1, e2)
print("Original vectors:\n e1 = \n", e1, "\n e2=\n", e2, "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)

utils.plot_transformation(utils.T_reflection_yaxis, e1, e2)

img = cv2.imread('images/leaf_original.png', 0)
plt.imshow(img)
plt.show()

image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_rotated)
plt.show()

rows, cols = image_rotated.shape
m = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image_rotated, m, (int(cols), int(rows)))
plt.imshow(image_rotated_sheared)
plt.show()

image_sheared = cv2.warpPerspective(img, m, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)
plt.show()

m_rotation_90_clockwise = np.array([[0, 1], [-1, 0]])
m_shear_x = np.array([[1, 0.5], [0, 1]])

print("90 degrees clockwise rotation matrix:\n", m_rotation_90_clockwise)
print("Matrix for the shear along x-axis:\n", m_shear_x)

print("m_rotation_90_clockwise by m_shear_x:\n", m_rotation_90_clockwise @ m_shear_x)
print("m_shear_x by m_rotation_90_clockwise:\n", m_shear_x @ m_rotation_90_clockwise)
