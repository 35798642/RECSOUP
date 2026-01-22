import torch
import numpy as np
import itertools
from scipy.spatial.distance import cdist

def batch_to_gpu(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}/{total} ({100*trainable/total:.2f}%)")

def sample_preference_vectors(num_vectors, num_objectives, alpha, device):
    rays = np.random.dirichlet([alpha] * num_objectives, num_vectors).astype(np.float32)
    return torch.tensor(rays, dtype=torch.float32).to(device)

def build_knn_graph(rays, k=3):
    dists = cdist(rays.detach().cpu().numpy(), rays.detach().cpu().numpy())
    edges = []
    for i in range(len(rays)):
        neighbors = np.argsort(dists[i])[1:k+1]
        for j in neighbors:
            edges.append((i, j))
    return edges

def compute_regularization(loss_matrix, edges):
    reg = 0.0
    T = loss_matrix.shape[1]
    eps = 1e-6
    for t in range(T):
        reg_t = 0.0
        for i, j in edges:
            diff = torch.abs(loss_matrix[i, t] - loss_matrix[j, t])
            reg_t += torch.exp(diff)
        reg_t = reg_t / len(edges)
        reg += torch.log(reg_t + eps)
    return reg

def generate_simplex_grid(n, m):
    """
    Generate a uniform grid of points on the n-dimensional simplex.

    Args:
        n (int): The dimension of the simplex.
        m (int): The number of grid points along each dimension.

    Returns:
        list: A list of n-dimensional vectors representing the grid points.
    """
    m = m - 1
    # **Generate all combinations of indices summing up to m**
    indices = list(itertools.combinations_with_replacement(range(m + 1), n - 1))

    # **Initialize an empty list to store the grid points**
    grid_points = []

    # **Iterate over each combination of indices**
    for idx in indices:
        # **Append 0 and m to the indices**
        extended_idx = [0] + list(idx) + [m]

        # **Compute the vector components by taking the differences between consecutive indices and dividing by m**
        point = [(extended_idx[i + 1] - extended_idx[i]) / m for i in range(n)]
        grid_points.append(point)

    return np.array(grid_points, dtype=np.float32)

