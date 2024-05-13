import numpy as np
import matplotlib.pyplot as plt
import time

def plot_vectors(list_v, list_label, list_color):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))
    
    
    plt.axis([-10, 10, -10, 10])
    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0]-0.2+sgn[0], v[1]-0.2+sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()

def dot(x, y):
    s = 0
    for xi, yi in zip(x, y):
        s += xi * yi
    return s

v = np.array([[1],[3]])
plot_vectors([v], [f"$v$"], ["black"])

plot_vectors([v, 2*v, -2*v], [f"$v$", f"$2v$", f"$-2v$"], ["black", "green", "blue"])

v = np.array([[1], [3]])
w = np.array([[4], [-1]])
plot_vectors([v, w, v + w], [f"$v$", f"$w$", f"$v + w$"], ["black", "black", "red"])
plot_vectors([v, w, np.add(v, w)], [f"$v$", f"$w$", f"$v + w$"], ["black", "black", "red"])

print("Norm of a vector v is", np.linalg.norm(v))

x = [1, -2, -5]
y = [4, 3, -1]
print("The dot product of x and y is", dot(x, y))
print("np.dot(x, y) function returns dot product of x and y:", np.dot(x, y))
print("This line output is a dot product of x and y: ", np.array(x) @ np.array(y))
print("\nThis line output is an error:")
try:
    print(x @ y)
except TypeError as err:
    print(err)

x = np.array(x)
y = np.array(y)

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = dot(a, b)
toc = time.time()
print("Dot product: ", c)
print("Time for the loop version: " + str(1000*(toc-tic)) + " ms")

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print("Dot product: ", c)
print("Time for the vectorized version, np.dot() version: " + str(1000*(toc-tic)) + " ms")

tic = time.time()
c = a @ b
toc = time.time()
print("Dot product: ", c)
print("Time for the vectorized version, @ function: " + str(1000*(toc-tic)) + " ms")

i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
print("The dot product of i and j is", dot(i, j))
