import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import *
from sympy.utilities.lambdify import lambdify
from jax import grad, vmap
import jax.numpy as jnp
import timeit, time

def f(x):
    ans = x**2
    print("FFFFFFFFF", ans)
    return x**2

def dfdx(x):
    return 2*x

def plot_f1_and_f2(f1, f2=None, x_min=-5, x_max=5, label1="f(x)", label2="f'(x)"):
    x = np.linspace(x_min, x_max,100)

    # Setting the axes at the centre.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, f1(x), 'r', label=label1)
    if not f2 is None:
        # If f2 is an array, it is passed as it is to be plotted as unlinked points.
        # If f2 is a function, f2(x) needs to be passed to plot it.        
        if isinstance(f2, np.ndarray):
            plt.plot(x, f2, 'bo', markersize=3, label=label2,)
        else:
            plt.plot(x, f2(x), 'b', label=label2)
    plt.legend()

    plt.show()

def f_composed(x):
    return np.exp(-2*x) + 3*np.sin(3*x)

def g(x):
#     return x**3
#     return 2*x**3 - 3*x**2 + 5
#     return 1/x
#     return jnp.exp(x)
#     return jnp.log(x)
#     return jnp.sin(x)
#     return jnp.cos(x)
    return jnp.abs(x)
#     return jnp.abs(x)+jnp.sin(x)*jnp.cos(x)

def f_polynomial_simple(x):
    return 2*x**3 - 3*x**2 + 5

def f_polynomial(x):
    for i in range(3):
        x = f_polynomial_simple(x)
    return x

print(f(3))
print(dfdx(3))

x_array = np.array([1, 2, 3])
print("x: \n", x_array)
print("f(x) = x**2: \n", f(x_array))
print("f'(x) = 2x: \n", dfdx(x_array))

plot_f1_and_f2(f, dfdx)

print(math.sqrt(18))
print(sqrt(18))
print(N(sqrt(18), 8))

x, y = symbols('x y')
expr = 2 * x**2 - x * y
print(expr)

expr_manip = x * (expr + x * y + x**3)
print(expr_manip)

print(expand(expr_manip))
print(factor(expr_manip))
print(expr.evalf(subs={x:-1, y:2}))

f_symb = x ** 2
print(f_symb.evalf(subs={x:3}))

print(x_array)

try:
    f_symb(x_array)
except TypeError as err:
    print(err)

f_symb_numpy = lambdify(x, f_symb, 'numpy')
print("x: \n", x_array)
print("f(x) = x**2: \n", f_symb_numpy(x_array))

print(diff(x**3, x))

dfdx_composed = diff(exp(-2*x) + 3*sin(3*x), x)
print(dfdx_composed)

dfdx_symb = diff(f_symb, x)
dfdx_symb_numpy = lambdify(x, dfdx_symb, 'numpy')

print("x: \n", x_array)
print("f'(x) = 2x: \n", dfdx_symb_numpy(x_array))

plot_f1_and_f2(f_symb_numpy, dfdx_symb_numpy)

dfdx_abs = diff(abs(x), x)
print(dfdx_abs)
print(dfdx_abs.evalf(subs={x:-2}))

dfdx_abs_numpy = lambdify(x, dfdx_abs, 'numpy')

try:
    dfdx_abs_numpy(np.array([1, -2, 0]))
except NameError as err:
    print(err)

x_array_2 = np.linspace(-5, 5, 100)
dfdx_numerical = np.gradient(f(x_array_2), x_array_2)

plot_f1_and_f2(dfdx_symb_numpy, dfdx_numerical, label1="f'(x) exact", label2="f'(x) approximate")
plot_f1_and_f2(lambdify(x, dfdx_composed, 'numpy'), np.gradient(f_composed(x_array_2), x_array_2), label1="f'(x) exact", label2="f'(x) approximate")

def dfdx_abs(x):
    if x > 0:
        return 1
    else:
        if x < 0:
            return -1
        else:
            return None

plot_f1_and_f2(np.vectorize(dfdx_abs), np.gradient(abs(x_array_2), x_array_2))

x_array_jnp = jnp.array([1., 2., 3.])
print("Type of NumPy array:", type(x_array))
print("Type of JAX NumPy array:", type(x_array_jnp))

x_array_jnp = jnp.array(x_array.astype('float32'))
print("JAX NumPy array:", x_array_jnp)
print("Type of JAX NumPy array:", type(x_array_jnp))

print(x_array_jnp * 2)
print(x_array_jnp[2])

try:
    x_array_jnp[2] = 4.0
except TypeError as err:
    print(err)

y_array_jnp = x_array_jnp.at[2].set(4.0)
print(y_array_jnp)
print(jnp.log(x_array))
print(jnp.log(x_array_jnp))

print("Function value at x = 3:", f(3.0))
print("Derivative value at x = 3:", grad(f)(3.0))

try:
    grad(f)(3)
except TypeError as err:
    print(err)

try:
    grad(f)(x_array_jnp)
except TypeError as err:
    print(err)

dfdx_jax_vmap = vmap(grad(f))(x_array_jnp)
print(dfdx_jax_vmap)

plot_f1_and_f2(f, vmap(grad(f)))
plot_f1_and_f2(g, vmap(grad(g)))

x_array_large = np.linspace(-5, 5, 1000000)

tic_symb = time.time()
res_symb = lambdify(x, diff(f(x), x), 'numpy')(x_array_large)
toc_symb = time.time()
time_symb = 1000 * (toc_symb - tic_symb)

tic_numerical = time.time()
res_numerical = np.gradient(f(x_array_large), x_array_large)
toc_numerical = time.time()
time_numerical = 1000 * (toc_numerical - tic_numerical)

tic_jax = time.time()
res_jax = vmap(grad(f))(jnp.array(x_array_large.astype('float32')))
toc_jax = time.time()
time_jax = 1000 * (toc_jax - tic_jax)

print(f"Results\nSymbolic Differentiation:\n{res_symb}\n" + 
      f"Numerical Differentiation:\n{res_numerical}\n" + 
      f"Automatic Differentiation:\n{res_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_symb} ms\n" + 
      f"Numerical Differentiation:\n{time_numerical} ms\n" + 
      f"Automatic Differentiation:\n{time_jax} ms")

tic_polynomial_symb = time.time()
res_polynomial_symb = lambdify(x, diff(f_polynomial(x), x), 'numpy')(x_array_large)
toc_polynomial_symb = time.time()
time_polynomial_symb = 1000 * (toc_polynomial_symb - tic_polynomial_symb)

tic_polynomial_jax = time.time()
res_polynomial_jax = vmap(grad(f_polynomial))(jnp.array(x_array_large.astype('float32')))
toc_polynomial_jax = time.time()
time_polynomial_jax = 1000 * (toc_polynomial_jax - tic_polynomial_jax)

print(f"Results\nSymbolic Differentiation:\n{res_polynomial_symb}\n" + 
      f"Automatic Differentiation:\n{res_polynomial_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_polynomial_symb} ms\n" +  
      f"Automatic Differentiation:\n{time_polynomial_jax} ms")
