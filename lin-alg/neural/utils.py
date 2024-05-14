import matplotlib.pyplot as plt
import numpy as np 

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
    plt.plot([0, e2[0][0], e1[0][0], e1[0][0]], 
             [0, e2[1][0], e2[1][0], e1[1][0]], 
             color=color_original)
    e1_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e1)])
    ax.text(e1[0]-0.2+e1_sgn[0], e1[1]-0.2+e1_sgn[1], f'$e_1$', fontsize=14, color=color_original)
    e2_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e2)])
    ax.text(e2[0]-0.2+e2_sgn[0], e2[1]-0.2+e2_sgn[1], f'$e_2$', fontsize=14, color=color_original)
    
    e1_transformed = T(e1)
    e2_transformed = T(e2)
    
    plt.quiver([0, 0],[0, 0], [e1_transformed[0], e2_transformed[0]], [e1_transformed[1], e2_transformed[1]], 
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0,e2_transformed[0][0], e1_transformed[0][0]+e2_transformed[0][0], e1_transformed[0][0]], 
             [0,e2_transformed[1][0], e1_transformed[1][0]+e2_transformed[1][0], e1_transformed[1][0]], 
             color=color_transformed)
    e1_transformed_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e1_transformed)])
    ax.text(e1_transformed[0][0]-0.2+e1_transformed_sgn[0][0], e1_transformed[1][0]-e1_transformed_sgn[1][0], 
            f'$T(e_1)$', fontsize=14, color=color_transformed)
    e2_transformed_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e2_transformed)])
    ax.text(e2_transformed[0][0]-0.2+e2_transformed_sgn[0][0], e2_transformed[1][0]-e2_transformed_sgn[1][0], 
            f'$T(e_2)$', fontsize=14, color=color_transformed)
    
    plt.gca().set_aspect("equal")
    plt.show()

def initialize_parameters(n_x):
    W = np.random.randn(1, n_x) * 0.01
    b = np.zeros((1, 1))
    
    assert (W.shape == (1, n_x))
    assert (b.shape == (1, 1))
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    return cost


def backward_propagation(A, X, Y):
    m = X.shape[1]

    dZ = A - Y
    dW = 1/m * np.matmul(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    W = parameters["W"]
    b = parameters["b"]
    
    dW = grads["dW"]
    db = grads["db"]
    
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

def train_nn(parameters, A, X, Y, learning_rate = 0.01):
    grads = backward_propagation(A, X, Y)
    parameters = update_parameters(parameters, grads, learning_rate)
    return parameters

def forward_propagation(x, parameters):
    w = parameters["W"]
    b = parameters["b"]
    
    z = w @ x + b
    y_hat = z

    return y_hat

def nn_model(x, y, num_iterations=1000, print_cost=False):
    n_x = x.shape[0]
    
    parameters = initialize_parameters(n_x) 
    
    for i in range(0, num_iterations):
        y_hat = forward_propagation(x, parameters)
        cost = compute_cost(y_hat, y)
        
        parameters = train_nn(parameters, y_hat, x, y, learning_rate = 0.001) 
        
        if print_cost:
            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(x, parameters):
    w = parameters['W']
    b = parameters['b']
    z = np.dot(w, x) + b
    return z

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

def transform_vectors(T, v1, v2):
    v = np.hstack((v1, v2))
    w = T(v)
    return w

def T_reflection_yaxis(v):
    a = np.array([[-1,0], [0,1]])
    w = a @ v
    return w

def T_stretch(a, v):
    t = np.array([[a,0], [0,a]])
    w = t @ v
    return w

def T_hshear(m, v):
    t = np.array([[1, m], [0, 1]])
    w = t @ v
    return w

def T_rotation(theta, v):
    t = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    w = t @ v
    return w

def T_rotation_and_stretch(theta, a, v):
    rotation_t = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    stretch_t = np.array([[a, 0], [0, a]])
    w = rotation_t @ (stretch_t @ v)
    return w
