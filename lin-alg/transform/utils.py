import numpy as np
import matplotlib.pyplot as plt

def T(v):
    w = np.zeros((3, 1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]
    return w

def L(v):
    a = np.array([[3,0], [0,0], [0,-2]])
    print("Transformation matrix:\n", a, "\n")
    w = a @ v
    return w

def T_hscaling(v):
    a = np.array([[2,0], [0,1]])
    w = a @ v
    return w

def T_reflection_yaxis(v):
    a = np.array([[-1,0], [0,1]])
    w = a @ v
    return w

def transform_vectors(T, v1, v2):
    v = np.hstack((v1, v2))
    w = T(v)
    return w

def plot_transformation(T, e1, e2):
    color_original = "#129cab"
    color_transformed = "#cc8933"
    
    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))
    
    plt.axis([-5, 5, -5, 5])
    plt.quiver([0, 0],[0, 0], [e1[0], e2[0]], [e1[1], e2[1]], color=color_original, angles='xy', scale_units='xy', scale=1)
    plt.plot([[0], e2[0], e1[0], e1[0]], 
             [[0], e2[1], e2[1], e1[1]], 
             color=color_original)
    e1_sgn = 0.4 * np.array([[1] if i==0 else i for i in np.sign(e1)])
    ax.text(e1[0]-0.2+e1_sgn[0], e1[1]-0.2+e1_sgn[1], f'$e_1$', fontsize=14, color=color_original)
    e2_sgn = 0.4 * np.array([[1] if i==0 else i for i in np.sign(e2)])
    ax.text(e2[0]-0.2+e2_sgn[0], e2[1]-0.2+e2_sgn[1], f'$e_2$', fontsize=14, color=color_original)
    
    e1_transformed = T(e1)
    e2_transformed = T(e2)
    
    plt.quiver([0, 0],[0, 0], [e1_transformed[0], e2_transformed[0]], [e1_transformed[1], e2_transformed[1]], 
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([[0],e2_transformed[0], e1_transformed[0]+e2_transformed[0], e1_transformed[0]], 
             [[0],e2_transformed[1], e1_transformed[1]+e2_transformed[1], e1_transformed[1]], 
             color=color_transformed)
    e1_transformed_sgn = 0.4 * np.array([[1] if i==0 else i for i in np.sign(e1_transformed)])
    ax.text(e1_transformed[0][0]-0.2+e1_transformed_sgn[0], e1_transformed[1][0]-e1_transformed_sgn[1][0], 
            f'$T(e_1)$', fontsize=14, color=color_transformed)
    e2_transformed_sgn = 0.4 * np.array([[1] if i==0 else i for i in np.sign(e2_transformed)])
    ax.text(e2_transformed[0][0]-0.2+e2_transformed_sgn[0][0], e2_transformed[1][0]-e2_transformed_sgn[1][0], 
            f'$T(e_2)$', fontsize=14, color=color_transformed)
    
    plt.gca().set_aspect("equal")
    plt.show()
