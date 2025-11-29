import numpy as np
import torch
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

features_path = 'platonic/results/features/Ozziey/poems_dataset/wit_1024/meta-llama_Llama-2-7b-chat-hf_pool-avg.pt'
features_path_2 = 'platonic/results/features/Ozziey/poems_dataset/wit_1024/mistralai_Mistral-7B-Instruct-v0.2_pool-avg.pt'
def load_radius_features(features_path: str = features_path, layer: int = -1) -> np.ndarray:
    """
    Load precomputed radius features from a .pt file.

    Args:
        features_path: Path to the .pt file containing the features.

    Returns:
        A NumPy array containing the loaded features.
    """
    features = torch.load(features_path)
    return features['feats'][:, layer, :].numpy()

def find_center_mean(features: np.ndarray) -> np.ndarray:
    """
    Find the center of the features in the feature space.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).

    Returns:
        A NumPy array representing the center of the features.
    """
    return np.mean(features, axis=0)


import numpy as np

def find_center_geometric_median(features: np.ndarray, eps: float = 1e-5, max_iter: int = 500) -> np.ndarray:
    """
    Find the geometric median of the features in the feature space using Weiszfeld's algorithm.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        eps: Convergence threshold.
        max_iter: Maximum number of iterations.

    Returns:
        A NumPy array representing the geometric median of the features.
    """
    # Initialize with the coordinate-wise median (robust starting point)
    median = np.median(features, axis=0)
    y = median.copy()
    
    for _ in range(max_iter):
        # Compute distances from current estimate
        distances = np.linalg.norm(features - y, axis=1)
        # Avoid division by zero
        nonzero = distances > eps
        if not np.any(nonzero):
            break
        inv_distances = 1.0 / distances[nonzero]
        weights = inv_distances / np.sum(inv_distances)
        y_new = np.sum(weights[:, np.newaxis] * features[nonzero], axis=0)
        
        # Check convergence
        if np.linalg.norm(y - y_new) < eps:
            return y_new
        y = y_new
    
    return y

def calculate_radius_max(features: np.ndarray, center: np.ndarray) -> float:
    """
    Calculate the radius of the features from the center.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        center: A NumPy array representing the center of the features.

    Returns:
        A float representing the radius of the features from the center.
    """
    distances = np.linalg.norm(features - center, axis=1)
    return np.max(distances)

def calculate_radius_two_sigma(features: np.ndarray, center: np.ndarray) -> float:
    """
    Calculate the radius of the features from the center, assuming a Gaussian distribution and using 2 standard deviations.
    The samples inside this radius should cover approximately within 2 sigma of the distribution.
    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        center: A NumPy array representing the center of the features.

    Returns:
        A float representing the radius of the features from the center.
    """
    distances = np.linalg.norm(features - center, axis=1)
    std_distance = np.std(distances)
    return 2 * std_distance


def get_radius(features: np.ndarray, method: str = 'two_sigma', center_method: str = 'mean') -> float:
    """
    Get the radius of the features using specified methods for center and radius calculation.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        method: Method to calculate radius ('max' or 'two_sigma').
        center_method: Method to calculate center ('mean' or 'geometric_median').
    Returns:
        A float representing the radius of the features.
    """
    if center_method == 'mean':
        center = find_center_mean(features)
    elif center_method == 'geometric_median':
        center = find_center_geometric_median(features)
    else:
        raise ValueError(f"Unknown center method: {center_method}")

    if method == 'max':
        radius = calculate_radius_max(features, center)
    elif method == 'two_sigma':
        radius = calculate_radius_two_sigma(features, center)
    else:
        raise ValueError(f"Unknown radius method: {method}")

    return radius

def gaussian_fit(features: np.ndarray) -> tuple:
    X = features
    N = X.shape[1]
    # Fit a 1-component Gaussian mixture model → equivalent to single multivariate Gaussian
    gmm = GaussianMixture(n_components=1, covariance_type='full')
    gmm.fit(X)
    means, cov = gmm.means_.flatten(), gmm.covariances_.flatten().reshape(N, N)
    print(means.shape, cov.shape)
    return means, cov

def gaussian_fit_independent(features: np.ndarray) -> tuple:
    X = features
    # Fit a 1-component Gaussian mixture model with diagonal covariance → independent dimensions
    gmm = GaussianMixture(n_components=1, covariance_type='diag')
    gmm.fit(X)

    return gmm.means_.flatten(), gmm.covariances_.flatten()

'''
features = load_radius_features(layer = -1)
radius = get_radius(features, method='max', center_method='mean')

# PCA visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

plt.scatter(features_2d[:, 0], features_2d[:, 1], s=1)
plt.title(f"PCA of Features (Radius: {radius})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# plot center

center = find_center_mean(features)
center_2d = pca.transform(center.reshape(1, -1))
plt.scatter(center_2d[0, 0], center_2d[0, 1], color='red', s=50, marker='x', label='Center')
plt.legend()

# plot circle of radius
circle = plt.Circle((center_2d[0, 0], center_2d[0, 1]), radius, color='green', fill=False, linestyle='--', label='Radius')
plt.gca().add_artist(circle)
plt.axis('equal')
plt.legend()
plt.xlim(features_2d[:, 0].min()-radius, features_2d[:, 0].max()+radius)
plt.ylim(features_2d[:, 1].min()-radius, features_2d[:, 1].max()+radius) 
plt.show()  
'''


def project_gaussian_to_2d(mean, cov):
    """
    Project an N-D Gaussian to 2D using PCA on its covariance.
    Returns projected_mean (2,), projected_cov (2,2)
    """

    # PCA projection matrix: N x 2
    pca = PCA(n_components=2)
    pca.fit(cov)  # PCA on covariance, not data

    W = pca.components_.T      # projection matrix (N x 2)
    mean_2d = W.T @ mean       # (2,)
    cov_2d = W.T @ cov @ W     # (2,2)

    return mean_2d, cov_2d


def draw_gaussian_ellipse(mean2d, cov2d, ax, nstd=2, **kwargs):
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nstd * np.sqrt(vals)

    ell = Ellipse(xy=mean2d, width=width, height=height,
                  angle=theta, **kwargs)
    ax.add_patch(ell)


features_1 = load_radius_features(layer = -1)
feature_2 = load_radius_features(features_path = features_path_2, layer = -1)
#print(gaussian_fit_independent(features))


mean2d, cov2d = project_gaussian_to_2d(*gaussian_fit(features_1))
mean2d_2, cov2d_2 = project_gaussian_to_2d(*gaussian_fit(feature_2))

# Plot the Gaussian ellipse
fig, ax = plt.subplots(figsize=(6,6))
draw_gaussian_ellipse(mean2d, cov2d, ax, nstd=2,
                      edgecolor='red', facecolor='none', linewidth=2)
draw_gaussian_ellipse(mean2d_2, cov2d_2, ax, nstd=2,
                      edgecolor='blue', facecolor='none', linewidth=2)

# Plot the mean points
ax.scatter(mean2d[0], mean2d[1], c='red', s=50)  # mean point
ax.scatter(mean2d_2[0], mean2d_2[1], c='blue', s=50)  # mean point
ax.set_title("Projected N-D Gaussian (GMM) to 2D")
ax.set_xlabel("PC 1 of Gaussian")
ax.set_ylabel("PC 2 of Gaussian")
ax.axis('equal')
plt.show()