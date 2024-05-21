import numpy as np
import glob
from matplotlib import image
import cv2
import matplotlib.pyplot as plt

def load_images(directory):
    images = []
    for filename in glob.glob(directory+'*.jpg'):
        img = np.array(image.imread(filename))
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gimg)

        height, width = gimg.shape
        
    return images

def plot_reduced_data(X):
    plt.figure(figsize=(12,12))
    plt.scatter(X[:,0], X[:,1], s=60, alpha=.5)
    for i in range(len(X)):
        plt.text(X[i,0], X[i,1], str(i),size=15)
    plt.show()

def check_eigenvector(p, x_inf):
    x_check = p @ x_inf
    return x_check

def center_data(y):
    mean_vector = np.mean(y, axis = 0)
    mean_matrix = np.repeat(mean_vector, y.shape[0])
    mean_matrix = np.reshape(mean_matrix, y.shape, order = 'F')

    x = y - mean_matrix
    return x

def get_cov_matrix(x):
    cov_matrix = np.transpose(x) @ x
    cov_matrix = cov_matrix / (x.shape[0] - 1)
    
    return cov_matrix

def perform_PCA(x, eigenvecs, k):
    v = eigenvecs[:,:k]
    xred = center_data(x) @ v

    return xred

def reconstruct_image(xred, eigenvecs):
    x_reconstructed = xred.dot(eigenvecs[:,:xred.shape[1]].T)

    return x_reconstructed
