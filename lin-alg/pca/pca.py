import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import utils
import tests

p = np.array([ 
    
    [0, 0.75, 0.35, 0.25, 0.85], 
    [0.15, 0, 0.35, 0.25, 0.05], 
    [0.15, 0.15, 0, 0.25, 0.05], 
    [0.15, 0.05, 0.05, 0, 0.05], 
    [0.55, 0.05, 0.25, 0.25, 0]  
]) 

x0 = np.array([[0],[0],[0],[1],[0]])
x1 = p @ x0

print(f'Sum of columns of P: {sum(p)}')
print(f'X1:\n{x1}')

tests.test_matrix(p, x0, x1)

x = np.array([[0], [0], [0], [1], [0]])
m = 20

for t in range(m):
    x = p @ x

print(x)

eigenvals, eigenvecs = np.linalg.eig(p)
print(f'Eigenvalues of P:\n{eigenvals}\n\nEigenvectors of P\n{eigenvecs}')

x_inf = eigenvecs[:,0]
print(f"Eigenvector corresponding to the eigenvalue 1:\n{x_inf[:,np.newaxis]}")

x_check = utils.check_eigenvector(p, x_inf)
print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(x_inf))
print("Result of multiplication:" + str(x_check))

print("Check that PX=X element by element:" + str(np.isclose(x_inf, x_check, rtol=1e-10)))

tests.test_check_eigenvector(utils.check_eigenvector)

x_inf = x_inf / sum(x_inf)
print(f"Long-run probabilities of being at each webpage:\n{x_inf[:,np.newaxis]}")

imgs = utils.load_images('./data/')
height, width = imgs[0].shape
print(f'\n Your dataset has {len(imgs)} images of size {height}x{width} pixels\n')

plt.imshow(imgs[0], cmap='gray')
plt.show()

imgs_flatten = np.array([im.reshape(-1) for im in imgs])
print(f'imgs_flatten shape: {imgs_flatten.shape}')

x = utils.center_data(imgs_flatten)
plt.imshow(x[0].reshape(64, 64), cmap='gray')
plt.show()

tests.test_center_data(utils.center_data)

cov_matrix = utils.get_cov_matrix(x)
print(f'Covariance matrix shape: {cov_matrix.shape}')

tests.test_cov_matrix(utils.get_cov_matrix)

np.random.seed(7)
eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k = 55)
print(f'Ten largest eigenvalues: \n{eigenvals[-10:]}')

eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[:,::-1]
eigenvecs = eigenvecs * -1
print(f'Ten largest eigenvalues: \n{eigenvals[:10]}')

fig, ax = plt.subplots(4, 4, figsize = (20, 20))
for n in range(4):
    for k in range(4):
        ax[n,k].imshow(eigenvecs[:,n*4+k].reshape(height, width), cmap='gray')
        ax[n,k].set_title(f'component number {n*4+k+1}')
plt.show()

xred2 = utils.perform_PCA(x, eigenvecs, 2)
print(f'xred2 shape: {xred2.shape}')

tests.test_check_PCA(utils.perform_PCA)

utils.plot_reduced_data(xred2)

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(imgs[2], cmap='gray')
ax[0].set_title('Image 2')
ax[1].imshow(imgs[16], cmap='gray')
ax[1].set_title('Image 16')
ax[2].imshow(imgs[15], cmap='gray')
ax[2].set_title('Image 15')
plt.suptitle('Similar cats')
plt.show()

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(imgs[12], cmap='gray')
ax[0].set_title('Image 12')
ax[1].imshow(imgs[15], cmap='gray')
ax[1].set_title('Image 15')
ax[2].imshow(imgs[7], cmap='gray')
ax[2].set_title('Image 7')
plt.suptitle('Different cats')
plt.show()

xred1 = utils.perform_PCA(x, eigenvecs,1)
xred5 = utils.perform_PCA(x, eigenvecs, 5)
xred10 = utils.perform_PCA(x, eigenvecs, 10)
xred20 = utils.perform_PCA(x, eigenvecs, 20)
xred30 = utils.perform_PCA(x, eigenvecs, 30)
xrec1 = utils.reconstruct_image(xred1, eigenvecs)
xrec5 = utils.reconstruct_image(xred5, eigenvecs)
xrec10 = utils.reconstruct_image(xred10, eigenvecs)
xrec20 = utils.reconstruct_image(xred20, eigenvecs)
xrec30 = utils.reconstruct_image(xred30, eigenvecs)

fig, ax = plt.subplots(2,3, figsize=(22,15))
ax[0,0].imshow(imgs[16], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(xrec1[16].reshape(height,width), cmap='gray')
ax[0,1].set_title('reconstructed from 1 components', size=20)
ax[0,2].imshow(xrec5[16].reshape(height,width), cmap='gray')
ax[0,2].set_title('reconstructed from 5 components', size=20)
ax[1,0].imshow(xrec10[16].reshape(height,width), cmap='gray')
ax[1,0].set_title('reconstructed from 10 components', size=20)
ax[1,1].imshow(xrec20[16].reshape(height,width), cmap='gray')
ax[1,1].set_title('reconstructed from 20 components', size=20)
ax[1,2].imshow(xrec30[16].reshape(height,width), cmap='gray')
ax[1,2].set_title('reconstructed from 30 components', size=20)
plt.show()

explained_variance = eigenvals/sum(eigenvals)
plt.plot(np.arange(1,56), explained_variance)
plt.show()

explained_cum_variance = np.cumsum(explained_variance)
plt.plot(np.arange(1,56), explained_cum_variance)
plt.axhline(y=0.95, color='r')
plt.show()

xred35 = utils.perform_PCA(x, eigenvecs, 35)
xrec35 = utils.reconstruct_image(xred35, eigenvecs)

fig, ax = plt.subplots(4,2, figsize=(15,28))
ax[0,0].imshow(imgs[28], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(xrec35[28].reshape(height, width), cmap='gray')
ax[0,1].set_title('Reconstructed', size=20)

ax[1,0].imshow(imgs[30], cmap='gray')
ax[1,0].set_title('original', size=20)
ax[1,1].imshow(xrec35[30].reshape(height, width), cmap='gray')
ax[1,1].set_title('Reconstructed', size=20)

ax[2,0].imshow(imgs[9], cmap='gray')
ax[2,0].set_title('original', size=20)
ax[2,1].imshow(xrec35[9].reshape(height, width), cmap='gray')
ax[2,1].set_title('Reconstructed', size=20)

ax[3,0].imshow(imgs[37], cmap='gray')
ax[3,0].set_title('original', size=20)
ax[3,1].imshow(xrec35[37].reshape(height, width), cmap='gray')
ax[3,1].set_title('Reconstructed', size=20)
plt.show()
