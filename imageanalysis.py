import matplotlib.pyplot as plt
import cv2
import os

import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, Ridge
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.linear_model import ElasticNet

import spgl1

image_number = 3
results_dir = "results/elasticnet"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# dimensionality (N) of subspace = 64
tile_w = 8
 
image = cv2.imread(f"Dataset/Image{image_number}.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
mindim = np.min(image.shape)
image = image[:mindim, :mindim]

plt.imshow(image)
plt.colorbar()
plt.savefig(os.path.join(results_dir, "image_%d.png"%(image_number)), bbox_inches='tight', pad_inches=0)
plt.close()

def return_tiles(image, tile_width):
    """
    image: A 2D array
    tile_width: the width of a square tile
    """
    width, height = image.shape

    # Calculate the number of tiles in each dimension
    num_tiles_x = width // tile_width
    num_tiles_y = height // tile_width

    # Initialize an empty array to store tiles
    # Reshape the image into tiles
    tiles = image[:num_tiles_y * tile_width, :num_tiles_x * tile_width].reshape(
        num_tiles_y, tile_width, num_tiles_x, tile_width)

    # Transpose the axes to get the desired shape
    tiles2d = tiles.transpose(0, 2, 1, 3).reshape(num_tiles_y, num_tiles_x, tile_width, tile_width)
    tiles1d = tiles2d.reshape(num_tiles_y*num_tiles_x, tile_width*tile_width)

    return tiles2d, tiles1d

im_tiles2d, im_tiles1d = return_tiles(image, tile_w)
print(im_tiles2d.shape)
print(im_tiles1d.shape)

def visualize_tiles(tiles_array):
    num_tiles_y, num_tiles_x, tile_width, _ = tiles_array.shape

    #subplot with a grid of tiles
    fig, axes = plt.subplots(num_tiles_y, num_tiles_x, figsize=(10, 10))

    # Iterate through each tile and display
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            axes[i, j].imshow(tiles_array[i, j, :, :], vmin=0, vmax=255)
            axes[i, j].axis('off')  # Turn off axis labels

    # plt.show()
    plt.savefig(os.path.join(results_dir, "image_%d_tiled.png"%(image_number)))
    plt.close()

visualize_tiles(im_tiles2d)
im_tiles1d = im_tiles1d.astype(float)


"""
Robust Subspace Clustering: Using GLMNet/ElasticNet
has beta_i = 0 constraint in a second way
https://glmnet.stanford.edu/articles/glmnet.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
"""
# Step 1: Compute sparse coefficients
Y = im_tiles1d
N, n = Y.shape
#3600,64
B = np.zeros((N, N))

output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'w')
for i in range(N):
    
    y_i = Y[i, :]
    
    # Remove the i-th row from Y
    y_others = np.delete(Y, i, axis=0)
    A = y_others.T  # Transpose to match dimensions (n, N-1)
    b = y_i.T       # (n,)

    # Set alpha to control regularization, l1_ratio for L1 regularization
    alpha = 5.8   # Adjust this value as needed
    l1_ratio = 1  # 1.0 gives Lasso (L1) regularization only

    # Initialize the ElasticNet model with non-negativity constraint
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False, max_iter=5000)
    enet.fit(A, b)
    result = enet.coef_

    if i % 10 == 0:
        print('%.d th tile result:' % (i), file=output_file)
        print('L1 norm b', np.linalg.norm(b, 1), file=output_file)
        print('L1 norm x', np.linalg.norm(result, 1), file=output_file)
        print('sum of x', np.sum(result), file=output_file)
        print('L2 norm Ax-b', np.linalg.norm(A @ result - b, 2), file=output_file)
    
    # Insert zero at the i-th position to maintain original dimensions
    beta_i = np.insert(result, i, 0)
    B[i, :] = np.abs(beta_i)

print("checking rough beta range:", B[1], file=output_file)
output_file.close()

# Step 2: Construct similarity graph
W = np.abs(B) + np.abs(B.T)

# Step 3: Compute degree matrix
D = np.diag(np.sum(W, axis=1))

# Step 4: Compute normalized Laplacian
D_sqrt_inv = np.linalg.inv(np.sqrt(D))
L_norm = D_sqrt_inv @ (D - W) @ D_sqrt_inv

# Step 5: Compute the eigenvalues
eigenvalues, _ = np.linalg.eigh(L_norm)
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
differences = np.diff(sorted_eigenvalues)
i_max = np.argmax(differences)
N = len(sorted_eigenvalues)
L_hat = N - (i_max + 1)

# To get the similarity matrix from the normalized Laplacian, we can use: S = I - L
S = np.eye(L_norm.shape[0]) - L_norm

# Step 6: Spectral clustering
spectral_clustering = SpectralClustering(n_clusters=L_hat, affinity='precomputed', random_state=0)
labels = spectral_clustering.fit_predict(W)

output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print('Extimated_Clusters:', L_hat, file=output_file)
print('shape of labels:',labels.shape, file=output_file)

print('determinant of W similarity matrix:', np.linalg.det(W), file=output_file)
output_file.close()

block_size = 128
block_cnt = int(im_tiles1d.shape[0]/block_size)
#8

#subplot with a grid of tiles
fig, axes = plt.subplots(block_cnt, block_cnt, figsize=(30, 30))

# Iterate through each tile and display
for i in range(block_cnt):
    for j in range(block_cnt):
        W_matrix = W[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        normalized_W_matrix = (W_matrix - np.min(W_matrix)) / (np.max(W_matrix) - np.min(W_matrix))
        axes[i, j].imshow(normalized_W_matrix, cmap='viridis', interpolation='none')
        axes[i, j].axis('off')  # Turn off axis labels

# plt.show()
plt.savefig(os.path.join(results_dir, "image_%d_similarities.png"%(image_number)))
plt.close()