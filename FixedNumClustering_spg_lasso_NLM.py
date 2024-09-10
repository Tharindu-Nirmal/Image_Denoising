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

image_number = 6
# dimensionality (N) of subspace = 64
tile_w = 8
step_size = 8 
std_dev = 100

results_dir = "results/FixedNum_spgl_lasso/tilw%d_step%d_noise%d"%(tile_w,step_size,std_dev)
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

    tau = 1
    # sigma = 100 #fixed sigma
    # sigma = 0.05 * np.linalg.norm(b, 2)
    sol_x, resid, grad, info = spgl1.spg_lasso(A, b,tau, verbosity=1)

    # print(type(sol_x)) --> <class 'numpy.ndarray'>
            
    result = np.array(sol_x).T
    # print(result.shape) # (3599,)

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

plt.savefig(os.path.join(results_dir, "image_%d_similarities.png"%(image_number)))
plt.close()

plt.figure(figsize=(12, 6))
patch_num_ex = 3
data = W[patch_num_ex,:]

#sorted values
top_count = 5
top_indices = np.argpartition(data, -top_count)[-top_count:]
top_indices = top_indices[np.argsort(data[top_indices])][::-1]

output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print("top %d similarities are in these indices in order"%(top_count) ,top_indices, file=output_file)
output_file.close()

plt.bar(range(len(data)), data)
plt.xlabel('patch number')
plt.ylabel('value in similarity graph')
plt.ylim(0, 1)
plt.title('variation in simillarity scores with the %d th patch'%(patch_num_ex))
plt.savefig(os.path.join(results_dir, "image_%d_similarity_w_%d th patch.png"%(image_number, patch_num_ex)))
plt.close()

plt.figure(figsize=(12, 6))
patch_num_ex = 7
data = W[patch_num_ex,:]

#sorted values
top_count = 5
top_indices = np.argpartition(data, -top_count)[-top_count:]
top_indices = top_indices[np.argsort(data[top_indices])][::-1]
output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print("top %d similarities are in these indices in order"%(patch_num_ex) ,top_indices, file=output_file)
output_file.close()


plt.bar(range(len(data)), data)
plt.xlabel('patch number')
plt.ylabel('value in similarity graph')
plt.ylim(0, 1)
plt.title('variation in simillarity scores with the %d th patch'%(patch_num_ex))
plt.savefig(os.path.join(results_dir, "image_%d_similarity_w_%d th patch.png"%(image_number, patch_num_ex)))
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
data1 = im_tiles1d[top_indices[0]]
data2 = im_tiles1d[top_indices[1]]
# data2 = im_tiles1d[15]
data3 = im_tiles1d[top_indices[2]]

# option 1
vmin = min(data1.min(), data2.min(), data3.min())
vmax = max(data1.max(), data2.max(), data3.max())

# option 2
# vmin = 0
# vmax = 255

img1 = axes[0].imshow(data1.reshape(tile_w,tile_w), cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title('Highest patch')

img2 = axes[1].imshow(data2.reshape(tile_w,tile_w), cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title('2nd highest patch')

img3 = axes[2].imshow(data3.reshape(tile_w,tile_w), cmap='viridis', vmin=vmin, vmax=vmax)
axes[2].set_title('3rd highest patch')

fig.colorbar(img1, ax=axes[0], orientation='vertical')
fig.colorbar(img2, ax=axes[1], orientation='vertical')
fig.colorbar(img3, ax=axes[2], orientation='vertical')
plt.savefig(os.path.join(results_dir, "image_%d_top3_similarity patches.png"%(image_number)))
plt.close()

nice_indices = [10,300,330]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
data1 = im_tiles1d[nice_indices[0]]
data2 = im_tiles1d[nice_indices[1]]
data3 = im_tiles1d[nice_indices[2]]

# option 1
# vmin = min(data1.min(), data2.min(), data3.min())
# vmax = max(data1.max(), data2.max(), data3.max())

# option 2
vmin = 0
vmax = 255

img1 = axes[0].imshow(data1.reshape(tile_w,tile_w), cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title('patch selection 1')

img2 = axes[1].imshow(data2.reshape(tile_w,tile_w), cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title('patch selection 2')

img3 = axes[2].imshow(data3.reshape(tile_w,tile_w), cmap='viridis', vmin=vmin, vmax=vmax)
axes[2].set_title('patch selection 3')

fig.colorbar(img1, ax=axes[0], orientation='vertical')
fig.colorbar(img2, ax=axes[1], orientation='vertical')
fig.colorbar(img3, ax=axes[2], orientation='vertical')
plt.savefig(os.path.join(results_dir, "image_%d_interesting patches.png"%(image_number)))
plt.close()



plt.figure(figsize=(12, 6))
patch_num_ex = 10
data = W[patch_num_ex,:]

#sorted values
top_count = 5
top_indices = np.argpartition(data, -top_count)[-top_count:]
top_indices = top_indices[np.argsort(data[top_indices])][::-1]
output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print("top %d similarities are in these indices in order"%(patch_num_ex) ,top_indices, file=output_file)
output_file.close()

plt.bar(range(len(data)), data)
plt.xlabel('patch number')
plt.ylabel('value in similarity graph')
plt.ylim(0, 1)
plt.title('variation in simillarity scores with the %d th patch'%(patch_num_ex))
plt.savefig(os.path.join(results_dir, "image_%d_simil_w_patch_%d.png"%(image_number, patch_num_ex)))
plt.close()



plt.figure(figsize=(12, 6))
patch_num_ex = 328
data = W[patch_num_ex,:]

#sorted values
top_count = 5
top_indices = np.argpartition(data, -top_count)[-top_count:]
top_indices = top_indices[np.argsort(data[top_indices])][::-1]
output_file = open(results_dir+'/image_%d prints.txt'%(image_number), 'a')
print("top %d similarities are in these indices in order"%(patch_num_ex) ,top_indices, file=output_file)
output_file.close()


plt.bar(range(len(data)), data)
plt.xlabel('patch number')
plt.ylabel('value in similarity graph')
plt.ylim(0, 1)
plt.title('variation in simillarity scores with the %d th patch'%(patch_num_ex))
plt.savefig(os.path.join(results_dir, "image_%d_simil_w_patch_%d.png"%(image_number, patch_num_ex)))
plt.close()

Spectral_cluster_indices = labels
num_clusters = L_hat

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
print ('keys of clustered_data:' ,clustered_data.keys())
print ('clustered_data[0] has shape' ,np.array(clustered_data[0]).shape)
print('cluster_medoids[0]  has shape', cluster_medoids[0].shape)
print ('cluster_means[0] has shape' , cluster_means[0].shape)

def get_centered_clusters(clustered_data, cluster_means):
    centered_clusters = {cluster: (np.array(clustered_data[cluster])-cluster_means[cluster]) for cluster in clustered_data.keys()}
    return centered_clusters

centered_clusters = get_centered_clusters(clustered_data, cluster_means)

#check for mean zero--> checked
summa = 0
for i in range(num_clusters):
    summa += centered_clusters[i].sum()
print (summa)

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
print(type(cluster_pca), cluster_pca.shape)
print(pca_vectors.shape)

num_clusters = len(centered_clusters)
# fig, axs = plt.subplots(nrows= num_clusters, ncols =1, figsize=(8,4* num_clusters))

#clusters_pcavectors and mean- a set basis(psi), one for each cluster
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

# Approximate x_hat = Psi_k alpha; for each tile x_hat
def fit_to_basis(data_vectors, basis_vectors):
    """ 
    basis_vectors : an nxN array with a basis vector(N-dimensional) in each row 
    data_vectors : an mxN array with m examples of (N-dimensional) data.
    """ 
    projection_matrix = basis_vectors.T @ (np.linalg.pinv(basis_vectors @ basis_vectors.T) @ basis_vectors)
    approximations = data_vectors @ projection_matrix
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