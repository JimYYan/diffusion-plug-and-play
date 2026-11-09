from functools import partial
import os
import argparse
import yaml
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt

import lpips
from skimage.metrics import peak_signal_noise_ratio

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, prepare_im
from util.logger import get_logger
from util.low_dim import generate_inverse_problem_gm, sliced_wasserstein
from dist import GaussMixTorch
from datetime import datetime


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    # Load diffusion sampler
    # print(use_warm_start)
    # sampler = create_sampler(adjust_color=adjust_color, warm_start=use_warm_start, **diffusion_config) 
    # sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn,
    #                     **diffusion_config.get('params', {}))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Working directory
    out_path = os.path.join(args.save_dir, datetime.now().strftime('%m-%d/%H:%M'), 'low-dim')
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Load GMM data
    def sample_gmm(n_samples, weights, means, covariances):
        """
        Sample from a Gaussian Mixture Model in PyTorch.

        Args:
            n_samples (int): number of samples to draw
            weights (Tensor): shape (K,) mixture weights (should sum to 1)
            means (Tensor): shape (K, D) means of Gaussians
            covariances (Tensor): shape (K, D, D) covariance matrices

        Returns:
            Tensor of shape (n_samples, D): the drawn samples
        """
        K, D = means.shape
        categorical = Categorical(weights)
        component_ids = categorical.sample((n_samples,))  # (n_samples,)

        # Prepare distributions
        mvns = [MultivariateNormal(means[k], covariances[k]) for k in range(K)]

        # Sample each point from its component
        samples = torch.stack([mvns[c].sample() for c in component_ids])
        return samples

    torch.manual_seed(100)
    dim = 2
    center = []
    scale = 1
    if dim == 1:
        tmp = 1
    else:
        tmp = 0
    for i in np.array((-1,1)):#range(-2, 3):
        center += [np.array([-8.*scale*i, -8.*scale*j]*(dim//2 + tmp)) for j in np.array((-1,1))]#range(-2, 3)]
    # for i in range(0,1):#range(-2, 3):
    #     center += [np.array([-8.*scale*i, -8.*scale*j]*(dim//2 + tmp)) for j in range(0,1)]

    weights = torch.rand((len(center),))**2
    weights = weights / weights.sum()
    center = np.array(center)
    center = torch.tensor(center)
    target = GaussMixTorch(center.float(), weights, scale)
    samples = target.sample_prior((1, 2000,))
    samples = samples.squeeze(dim=0)
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
    plt.title("Samples from GMM")
    plt.axis('equal')
    plt.savefig("./fig/prior_gmm.png")
    plt.close()

    dim = 2
    dim_y = 2
    snr_range = torch.tensor([1e-1, 1e-1])
    kappa_range = torch.tensor([20, 20])
    y, A, Sigma_y, x_origin = generate_inverse_problem_gm(dim, dim_y, target, scale, kappa_range, snr_range)
    var_y = Sigma_y[0][0]
    U, D, V = torch.linalg.svd(A, full_matrices=False)
    V = V.T
    print(var_y)

    class OperatorLowDim():
        def __init__(self, A) -> None:
            self.A = A

        def forward(self, x, **kwargs):
            return (A @ x.T).T

    operator_lowdim = OperatorLowDim(A)    
    samples_true_full = target.sample_posterior((1, 2000,))
    samples_true_full = samples_true_full.squeeze(dim=0)
    plt.scatter(samples_true_full[:, 0], samples_true_full[:, 1], s=5, alpha=0.5)
    plt.title("Samples from GMM")
    plt.axis('equal')
    plt.savefig("./fig/posterior_gmm.png")
    plt.close()

    # Do Inference
    x0s = torch.randn_like(samples_true_full)
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    noiser = get_noise('gaussian', sigma = var_y**0.5)
    cond_method = get_conditioning_method('mala', operator_lowdim, noiser, scale=0.5)
    measurement_cond_fn = cond_method.conditioning
    default_cond_method = cond_method
    default_cond_fn = default_cond_method.conditioning

    use_warm_start = 'initializer' in cond_config
    if use_warm_start:
        initializer = cond_config['initializer']
        default_cond_method = get_conditioning_method(
            initializer['method'], operator_lowdim, noiser, **initializer['params'])
    else:
        default_cond_method = cond_method
    default_cond_fn = default_cond_method.conditioning

    # load model
    def model(x, t):
        # log_prob = torch.log(torch.exp(-torch.sum((x - center*torch.exp(-t))**2, dim=-1)/2).mean(dim=0))
        # score = torch.autograd.grad(log_prob, x)[0]
        return target.score_fn(x, t)
    
    # Load diffusion sampler
    sampler = create_sampler(adjust_color=False, warm_start=False, **diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn,
                        **diffusion_config.get('params', {}))
    
    for i, x0 in enumerate(x0s):
        # fname = str(i).zfill(5) + diffusion_config['sampler'] + '.png'
        # ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        # if measure_config['operator'] ['name'] == 'inpainting':
        #     mask = mask_gen(ref_img)
        #     mask = mask[:, 0, :, :].unsqueeze(dim=1)
        measurement_cond_fn = partial(cond_method.conditioning, mask=A)
        default_cond_fn = partial(default_cond_method.conditioning, mask=A)
        sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
         
        # Sampling
        # if 'init_config' in data_config:
        #     x_start = next(initializer).to(device)
        # else:

        # x_start = torch.randn_like(x0.unsqueeze(0)).requires_grad_()
        # y = y.unsqueeze(0)
        x_start = torch.randn_like(samples_true_full).requires_grad_()
        x_start = x_start * 10
        # x_start = samples_true_full.clone().requires_grad_()
        y = y.unsqueeze(0).repeat(samples_true_full.shape[0], 1)

        sample = sample_fn(
            x_start=x_start, 
            measurement=y, 
            default_cond_fn=default_cond_fn, 
            record=True, 
            save_root=out_path
        )
        # samples.append(sample.detach().cpu().numpy())
        break

    # samples = np.array(samples)
    samples = sample.detach().cpu().numpy()
    plt.scatter(samples[:,0], samples[:,1], s=5, alpha=0.5, c='b', label='recon')
    plt.scatter(samples_true_full[:, 0], samples_true_full[:, 1], s=5, alpha=0.5, c='r', label='gt')
    plt.title("Samples from GMM")
    plt.legend()
    plt.axis('equal')
    plt.savefig("./fig/recon_gmm.png")
    plt.close()



if __name__ == '__main__':
    main()
