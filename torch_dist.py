import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

class Distribution:
    def __init__(self, sigma0=1.0):
        self.sigma0 = sigma0
        self.predictor_fn = lambda z, t: torch.exp(t) * z + (torch.exp(t) - torch.exp(-t)) * self.score_fn(z, t)
        self.rt2 = lambda t: ((torch.exp(2 * t) - 1) * self.sigma0 ** 2) / ((torch.exp(2 * t) - 1) + self.sigma0 ** 2)

    def score_fn(self, z, t):
        raise NotImplementedError("score_fn must be implemented in a subclass")

    def logposterior_density(self, z, t, y, posterior_dist):
        logits = posterior_dist.mixture_distribution.logits
        centers = posterior_dist.component_distribution.loc
        variances = posterior_dist.component_distribution.variance

        exp_neg2t = torch.exp(-2 * t)
        denom = variances * exp_neg2t + (1 - exp_neg2t)

        diff = z.unsqueeze(-2) - centers * torch.exp(-t)
        quad = -torch.sum(diff ** 2 / denom, dim=-1) / 2
        log_weights = F.log_softmax(logits, dim=-1)
        component_log_probs = quad + log_weights
        return torch.logsumexp(component_log_probs, dim=-1)

    def sample_posterior(self, sample_shape, posterior_dist):
        return posterior_dist.sample(sample_shape)

def make_posterior_mixture(logits, means, covariances):
    """
    Construct a MixtureSameFamily posterior from logits, means, and covariances.

    Args:
        logits: Tensor of shape (K,)
        means: Tensor of shape (K, D)
        covariances: Tensor of shape (K, D, D)

    Returns:
        MixtureSameFamily distribution object
    """
    mix = Categorical(logits=logits)
    components = MultivariateNormal(loc=means, covariance_matrix=covariances)
    return MixtureSameFamily(mix, components)
