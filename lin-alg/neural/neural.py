import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import tests

v = np.array([[3], [5]])
w = utils.T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

u = np.array([[1], [-2]])
v = np.array([[2], [4]])
k = 7

print("T(k*v):\n", utils.T(k*v), "\n k*T(v):\n", k*utils.T(v), "\n\n")
print("T(u+v):\n", utils.T(u+v), "\n\n T(u)+T(v):\n", utils.T(u)+utils.T(v))

v = np.array([[3], [5]])
w = utils.L(v)
print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

img = np.loadtxt('data/image.txt')
print("Shape: ", img.shape)
print(img)
plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.show()

e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])
transformation_result_hscaling = utils.transform_vectors(utils.T_hscaling, e1, e2)
print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

utils.plot_transformation(utils.T_hscaling, e1, e2)

plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.scatter(utils.T_hscaling(img)[0], utils.T_hscaling(img)[1], s = 0.001, color = 'grey')
plt.show()

e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])
transformation_result_reflection_yaxis = utils.transform_vectors(utils.T_reflection_yaxis, e1, e2)
print("Original vectors:\n e1= \n", e1,"\n e2=\n", e2, "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)

utils.plot_transformation(utils.T_reflection_yaxis, e1, e2)

plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.scatter(utils.T_reflection_yaxis(img)[0], utils.T_reflection_yaxis(img)[1], s = 0.001, color = 'grey')
plt.show()

tests.test_T_stretch(utils.T_stretch)

plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.scatter(utils.T_stretch(2, img)[0], utils.T_stretch(2, img)[1], s = 0.001, color = 'grey')
plt.show()

utils.plot_transformation(lambda v: utils.T_stretch(2, v), e1, e2)

tests.test_T_hshear(utils.T_hshear)

plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.scatter(utils.T_hshear(2, img)[0], utils.T_hshear(2, img)[1], s = 0.001, color = 'grey')
plt.show()

utils.plot_transformation(lambda v: utils.T_hshear(2, v), e1, e2)

tests.test_T_rotation(utils.T_rotation)

plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.scatter(utils.T_rotation(np.pi, img)[0], utils.T_rotation(np.pi, img)[1], s = 0.001, color = 'grey')
plt.show()

utils.plot_transformation(lambda v: utils.T_rotation(np.pi, v), e1, e2)

tests.test_T_rotation_and_stretch(utils.T_rotation_and_stretch)

plt.scatter(img[0], img[1], s = 0.001, color = 'black')
plt.scatter(utils.T_rotation_and_stretch(np.pi, 2, img)[0], utils.T_rotation_and_stretch(np.pi, 2, img)[1], s = 0.001, color = 'grey')
plt.show()

utils.plot_transformation(lambda v: utils.T_rotation_and_stretch(np.pi, 2, v), e1, e2)

parameters = utils.initialize_parameters(2)
print(parameters)

tests.test_forward_propagation(utils.forward_propagation)

tests.test_nn_model(utils.nn_model)

df = pd.read_csv("data/toy_dataset.csv")
print(df.head())

x = np.array(df[['x1', 'x2']]).T
y = np.array(df['y']).reshape(1, -1)

parameters = utils.nn_model(x, y, num_iterations = 5000, print_cost = True)
y_hat = utils.predict(x, parameters)
df['y_hat'] = y_hat[0]

for i in range(10):
    print(f"(x1,x2) = ({df.loc[i, 'x1']:0.2f}, {df.loc[i, 'x2']:0.2f}): Actual value: {df.loc[i, 'y']:0.2f}. Predicted value: {df.loc[i, 'y_hat']:0.2f}")
