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
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import ElasticNet
import spgl1

image_number = 3
# dimensionality (N) of subspace = 64
tile_w = 8
step_size = 8 
std_dev = 50

results_dir = "results/NC_elasticnet_meanenf/tilw%d_step%d_noise%d"%(tile_w,step_size,std_dev)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

 
image = cv2.imread(f"Dataset/Image{image_number}.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
mindim = np.min(image.shape)
mindim = int((mindim // tile_w) * tile_w)
image = image[:mindim, :mindim]
print("image shape: ", image.shape)

plt.imshow(image)
plt.colorbar()
plt.savefig(os.path.join(results_dir, "image_%d.png"%(image_number)), bbox_inches='tight', pad_inches=0)
plt.close()


noisy_image = np.uint8(np.clip(image + np.random.normal(scale=std_dev, size=image.shape), 0, 255))
plt.imshow(noisy_image)
plt.colorbar()
plt.savefig(os.path.join(results_dir, "image_%d.png"%(image_number)), bbox_inches='tight', pad_inches=0)
plt.close()

def return_overlapping_tiles(image, tile_width, step_size):
    """
    image: A 2D array
    tile_width: the width of a square tile
    step_size: the number of pixels to move the window in both directions
    
    Returns:
    - tiles1d: Flattened patches from the image.
    - patch_count: 2D array where each element counts how many patches cover each pixel.
    """
    height, width = image.shape

    # Initialize an empty list to store the flattened patches
    tiles1d = []

    # Initialize a 2D array to count how many patches each pixel is part of
    patch_count = np.zeros_like(image, dtype=int)

    # Create overlapping tiles using a sliding window approach with step size
    for i in range(0, height - tile_width + 1, step_size):
        for j in range(0, width - tile_width + 1, step_size):
            # Extract the tile
            tile = image[i:i+tile_width, j:j+tile_width]
            
            # Flatten the tile and add to the list
            tiles1d.append(tile.flatten())

            # Increment the patch count for each pixel in the tile
            patch_count[i:i+tile_width, j:j+tile_width] += 1

    # Convert the list of tiles to a numpy array
    tiles1d = np.array(tiles1d)

    return tiles1d, patch_count

def reconstruct_image(imtiles1d, patch_count, tile_width, original_shape, step_size):
    """
    Reconstructs the 2D image from the processed imtiles1d patches.
    
    imtiles1d: The processed patches, same shape as original tiles1d.
    patch_count: 2D array with the count of patches each pixel is involved in.
    tile_width: Width of the square tile.
    original_shape: Shape of the original image (height, width).
    step_size: The number of pixels the window moves by in both directions.
    """
    height, width = original_shape

    # Initialize an empty array for the reconstructed image
    reconstructed_image = np.zeros((height, width), dtype=float)

    # Recreate the sliding window effect with step size
    patch_idx = 0
    for i in range(0, height - tile_width + 1, step_size):
        for j in range(0, width - tile_width + 1, step_size):
            # Reshape the 1D patch back to a 2D tile
            patch = imtiles1d[patch_idx].reshape(tile_width, tile_width)

            # Accumulate the patch values back into the reconstructed image
            reconstructed_image[i:i+tile_width, j:j+tile_width] += patch

            # Move to the next patch
            patch_idx += 1

    # Normalize the reconstructed image by dividing by patch_count to handle overlapping regions
    reconstructed_image /= patch_count

    return reconstructed_image

# The data vector
im_tiles1d, patch_count = return_overlapping_tiles(noisy_image, tile_w, step_size)
print(im_tiles1d.shape)
print(patch_count.shape)
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
output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')

for i in range(N):
    
    y_i = Y[i, :]
    
    # Remove the i-th row from Y
    y_others = np.delete(Y, i, axis=0)
    A = y_others.T  # Transpose to match dimensions (n, N-1)
    b = y_i.T       # (n,)

     # Set alpha to control regularization, l1_ratio for L1 regularization
    alpha = 10   # Adjust this value as needed
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

#We fix this since the algorithms' value just  blows up
L_hat = 20 #N - (i_max + 1)

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

Spectral_cluster_indices = labels
num_clusters = L_hat
#Clustering is done................................................................................

def calculate_medoid(cluster):
    """
    cluster: an mxN array of m vectors in a cluster
    """
    # Pairwise distances between points in the cluster
    distances = cdist(cluster, cluster, metric='euclidean')
    total_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(total_distances)

    # Medoid is the data point with the minimum total distance
    medoid = cluster[medoid_index]

    return medoid

def get_cluster_medoids(data, cluster_indices):
    """
    Inputs:
    data: an mxN array of m data vectors (points)
    cluster_indices: an mx1 array of m cluster indices

    Returns:
    clusters: a dictionary, indexed by cluster number and values are clustered arrays of data
    medoids: a dictionary, indexed by cluster number and values are the medoid data vector
    means: a dictionary, indexed by cluster number and values are means
    
    """
    unique_clusters = np.unique(cluster_indices)

    # Initialise a dictionary to store clusters
    clusters = {cluster: [] for cluster in unique_clusters}

    # Populate clusters dictionary with data points
    for i, cluster_index in enumerate(cluster_indices):
        clusters[cluster_index].append(data[i])

    # Calculate medoid for each cluster
    medoids = {cluster: calculate_medoid(np.array(points)) for cluster, points in clusters.items()}
    means = {cluster: np.mean(np.array(points), axis=0) for cluster, points in clusters.items()}

    return clusters, medoids, means

clustered_data, cluster_medoids, cluster_means = get_cluster_medoids(im_tiles1d, Spectral_cluster_indices)

output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print ('keys of clustered_data:' ,clustered_data.keys(), file=output_file)
print ('clustered_data[0] has shape' ,np.array(clustered_data[0]).shape, file=output_file)
print('cluster_medoids[0]  has shape', cluster_medoids[0].shape, file=output_file)
print ('cluster_means[0] has shape' , cluster_means[0].shape, file=output_file)
output_file.close()

def get_centered_clusters(clustered_data, cluster_means):
    centered_clusters = {cluster: (np.array(clustered_data[cluster])-cluster_means[cluster]) for cluster in clustered_data.keys()}
    return centered_clusters

centered_clusters = get_centered_clusters(clustered_data, cluster_means)

# #check for mean zero--> checked
# summa = 0
# for i in range(num_clusters):
#     summa += centered_clusters[i].sum()

# output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
# print ('confirming centred clusters',summa,file=output_file)
# output_file.close()

def pca_for_cluster(cluster):
    """ 
    Inputs:
    cluster- mxN centered array of vectors from a single cluster

    Returns:
    cluster_pca
    """
    assert isinstance(cluster, np.ndarray)

    pca = PCA()
    cluster_pca = pca.fit_transform(cluster)
    pca_vectors = pca.components_
    # print('pca vectors shape:', pca_vectors.shape)


    #padding to make the cumsum array have data_dim(8x8=64) length
    data_dim = cluster.shape[-1] 
    padding_size = max(0, data_dim - len(pca.explained_variance_ratio_))
    expl_var_ratio_cumul = np.cumsum(np.pad(pca.explained_variance_ratio_, (0, padding_size), 'constant', constant_values=0))

    return cluster_pca, expl_var_ratio_cumul, pca_vectors

#test pca function
cluster_pca, expln_var_cum, pca_vectors = pca_for_cluster(centered_clusters[0])
# print('cluster [0] type,shape: ',type(cluster_pca), cluster_pca.shape)
# print(pca_vectors.shape)

num_clusters = len(centered_clusters)

#clusters_pca vectors and mean- a set basis(psi), one for each cluster
t_exp = 0.9
dynamic_psi = dict()
fixed_psi = dict()
dim_comp = 0.5
fixed_cut = int(dim_comp* np.square(tile_w))

for i, (cluster_key, points) in enumerate(centered_clusters.items()):
    cluster_pca, expln_var_cum, pca_vectors = pca_for_cluster(points)

    # find the index that reaches 0.9 cumsum variability
    cutidx = np.argmax(expln_var_cum >= t_exp)
    dynamic_basis = np.vstack((np.array([cluster_means[cluster_key]]), pca_vectors[:cutidx]))
    fixed_basis = np.vstack((np.array([cluster_means[cluster_key]]), pca_vectors[:fixed_cut]))

    # Add the dynamic_basis vectors describing the 0.9 variance
    dynamic_psi[cluster_key] = dynamic_basis

    # Add the fixed_basis vectors which are the 50% of the the basis vectors
    fixed_psi[cluster_key] = fixed_basis

    print('Cluster %d dynamic_basis vectors shape:'%(cluster_key), dynamic_basis.shape)
    print('Cluster compression when pruning %.2f variance is %.4f'%(t_exp, dynamic_basis.shape[0]/points.shape[0]))
    print('--------------------')

# Approximate x_hat = Psi_k alpha; for each tile x_hat, enforcing I have mean coefficient=1
def fit_to_basis(data_vectors, basis_vectors):
    """ 
    basis_vectors : an nxN array with a basis vector(N-dimensional) in each row 
    data_vectors : an mxN array with m examples of (N-dimensional) data.
    """ 
    mean_vector = basis_vectors[0]   # The first row is the mean vector
    pca_vectors = basis_vectors[1:]  # The remaining rows are the PCA vectors

    projection_matrix = pca_vectors.T @ (np.linalg.pinv(pca_vectors @ pca_vectors.T) @ pca_vectors)
    pca_projection = data_vectors @ projection_matrix

    approximations = mean_vector + pca_projection
    errors = np.linalg.norm(data_vectors - approximations , axis=1)
    # print(errors.shape)
    return approximations, errors

dyn_errors = []
fix_errors = []
for i, (cluster_key, points) in enumerate(clustered_data.items()):
    dyn_approx, dyn_errs = fit_to_basis(clustered_data[cluster_key], dynamic_psi[cluster_key])
    fix_approx, fix_errs = fit_to_basis(clustered_data[cluster_key], fixed_psi[cluster_key])
    dyn_errors.append(np.mean(dyn_errs))
    fix_errors.append(np.mean(fix_errs))
    print(" cluster %d has error after fitting: \n dynamic basis selection: %.4f \n fixed top %.2f: %.4f \n ------ "%(i, np.mean(dyn_errs), dim_comp, np.mean(fix_errs)))

# print(np.array(dyn_errors).shape)

fig, axs = plt.subplots(nrows= 2, ncols =1, figsize=(8,10))
axs[0].bar(np.array(range(num_clusters)), dyn_errors, label='Dynamic selection of cumul.variance %.2f pca vectors'%(t_exp))
axs[0].set_xlabel('Cluster number')
axs[0].set_ylabel('Mean L2 error within the clustered data')
axs[0].legend()

axs[1].bar(range(num_clusters), fix_errors, label='Fixed selection of top %.2f pca vectors'%(dim_comp))
axs[1].set_xlabel('Cluster number')
axs[1].set_ylabel('Mean L2 error within the clustered data')
axs[1].legend()
plt.savefig(os.path.join(results_dir, "image_%d_Choice_Basis.png"%(image_number)))
plt.close()

print('1D data array shape:',im_tiles1d.shape)
print('cluster indices shape',Spectral_cluster_indices.shape)

def visualise_approx(im_tiles1d, cluster_indices):
    approx_data1d = np.zeros_like(im_tiles1d)
    error_data = np.zeros_like(im_tiles1d)
    for i in range(len(cluster_indices)):
        fix_approx, fix_errs = fit_to_basis(im_tiles1d[i][np.newaxis,:],fixed_psi[cluster_indices[i]])
        approx_data1d[i] = fix_approx
        error_data[i] = fix_errs

    num_tiles_x = mindim
    num_tiles_y = mindim

    original_shape = num_tiles_x, num_tiles_y

    tile_width = tile_w
    tile_height = tile_w

    approx_image = reconstruct_image(approx_data1d, patch_count, tile_width, original_shape, step_size)

    return approx_image

approx_image = visualise_approx(im_tiles1d, Spectral_cluster_indices)

plt.imshow(approx_image)
plt.colorbar()
plt.savefig(os.path.join(results_dir, "image_%d_ApproxImage.png"%(image_number)))
plt.close()

plt.imshow(approx_image, cmap='gray', vmin=0, vmax=255)  # Display in grayscale
plt.axis('off')  # Turn off axis labels if desired
plt.savefig(os.path.join(results_dir, "image_%d_ApproxImage_Gray.png" % image_number), bbox_inches='tight', pad_inches=0)  # Save without colorbar
plt.close()

MSE = np.mean(np.square(approx_image.astype(np.float32) - image.astype(np.float32)))
PSNR =cv2.PSNR(image.astype(np.float32), approx_image.astype(np.float32))
ssim_value, ssim_map = ssim(approx_image.astype(np.float32), image.astype(np.float32), full=True)

output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print('MSE:', MSE, file=output_file)
print('PSNR:', PSNR, file=output_file)
print('SSIM:', ssim_value, file=output_file)
output_file.close()


plt.imshow(ssim_map)
plt.colorbar()
plt.savefig(os.path.join(results_dir, "image_%d_SSIM_Map.png"%(image_number)))
plt.close()