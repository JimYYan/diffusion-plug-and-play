import time
import numpy as np
import torch
from functools import partial
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def generate_inverse_problem_gm(dim, dim_y, target, scale=1, kappa_range=None, snr_range=None):
    A = torch.randn((dim_y, dim))
    U, D, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = np.ones_like(V[0])
    coordinate_mask[len(D):] = 0
    
    if kappa_range is not None or snr_range is not None:
        assert dim_y == dim, "Only support non-degegenerate observation for setting kappa or SNR; otherwise, by definition kappa = infty and SNR=0 since lambda_min=0"
    if kappa_range is None:
        diag = torch.sort(torch.rand(D.shape) * -1) * -1
    else:
        kappa = kappa_range[0]#, high=kappa_range[1], size=(1,))[0]
        # diag = torch.sort(torch.randint(low=1/kappa, high=1.0, size=D.shape) * -1) * -1
        diag, _ = torch.sort(torch.empty(D.shape).uniform_(1/kappa, 1.0) * -1)
        diag = list(diag * -1)
        diag[0] = 1.0
        diag[-1] = 1/kappa
        diag = torch.tensor(diag)
    A = U @ (torch.diag(diag) @ V[coordinate_mask == 1, :]) / scale
    # A = jnp.concatenate([U, jnp.zeros((dim_y, dim - dim_y))], axis=1) # for orthogonal A
    # A = jnp.concatenate([jnp.eye(dim_y), jnp.zeros((dim_y, dim - dim_y))], axis=1)
    init_sample = target.prior_dist.sample(sample_shape=())
    
    if snr_range is None:
        var_observations = torch.randint(size=(1,), low=0.2, high=1.0)[0] * torch.ones(len(diag)) * max(diag)**2
        # var_observations = random.uniform(subkey, (1,), minval=0.05, maxval=0.06)[0] * jnp.ones(len(diag)) * max(diag)**2
    else:
        snr = snr_range[0]
        var_observations = torch.ones(len(diag)) * min(diag)**2 / snr
    std = var_observations**0.5
    # var_observations = std**2
    if dim == 1:
        init_obs = A * init_sample
    else:
        init_obs = A @ init_sample
   
    init_obs += torch.randn(init_obs.shape) * (std)
    Sigma_y = torch.diag(var_observations)
    target.set_observation(init_obs, A, Sigma_y)
    return init_obs, A, Sigma_y, init_sample

def sliced_wasserstein(sample1, sample2, n_slices=100):
    projections = torch.randn(shape=(n_slices, sample1.shape[1]))
    projections = projections / torch.linalg.norm(projections, axis=-1, keepdims=True)
    sample1_projected = (projections @ sample1.T)
    sample2_projected = (projections @ sample2.T)
    return np.mean([wasserstein_distance(u_values=s1, v_values=s2) for s1, s2 in zip(sample1_projected, sample2_projected)])