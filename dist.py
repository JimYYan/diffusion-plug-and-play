import math
import numpy as np
import torch
import numpyro
# import jax
# from jax import grad
import torchvision.transforms as transforms
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
# from jax import random
# from jax import grad
# from numpyro.distributions import MultivariateNormal, Categorical, MixtureSameFamily

class Distribution:
    def __init__(self):
        # derived from the Tweedie's Formula, asscociated with standard OU process: E[e^{-t} X_0] = X_t + \sqrt{1-e^{-2t}} * score
        self.predictor_fn = lambda z, t: torch.exp(t)*z + (torch.exp(t) - torch.exp(-t)) * self.score_fn(z, t)
        # assume prior covariance is proportional to identity, with value sigma0 ** 2
        # self.rt2 = lambda t: ((1 - jnp.exp(-2*t)) * self.sigma0 ** 2) / ((1 - jnp.exp(-2*t)) + self.sigma0 ** 2) # this one is incorrect, but better for high-dim examples without resampling
        # not needed for general prior covariance
        self.rt2 = lambda t: ((torch.exp(2*t) - 1) * self.sigma0 ** 2) / ((torch.exp(2*t) - 1) + self.sigma0 ** 2)
    
    def score_fn(self, z, t):
        raise NotImplementedError
        
    def posterior_score_fn(self, z, t, y):
        raise NotImplementedError
        
    def sample_prior(self, n):
        raise NotImplementedError
        
    def sample_posterior(self, n, y):
        raise NotImplementedError


class StandNorm(Distribution):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.sigma0 = 1
        # sanity check
        t = 0.2
        z = torch.randn(1, dim); y = torch.randn(1, dim);
        assert np.allclose(self.posterior_score_fn(z, t, y), self.pseudo_cond_score_fn(z, t, y)+self.score_fn(z, y), 1e-4), "Sanity check wrong"
        
    def score_fn(self, z, t):
        return -z
    
    def sample_prior(self, n):
        return torch.randn(n, self.dim)
    
    def posterior_score_fn(self, z, t, y):
        return -(z - torch.exp(-t)*y/2) / (1-torch.exp(-2*t)/2)
    
    def sample_posterior(self, key, n, y):
        return torch.randn(n, self.dim) * torch.sqrt(1/2) + y/2

class GaussMix(Distribution):
    def __init__(self, center):
        super().__init__()
        # assume weights are equal
        self.dim = center.shape[1]
        self.center = center # of shape n_cetern * dim
        self.n_center = center.shape[0]
        self.sigma0 = torch.sqrt(torch.mean(center ** 2, axis=0) + 1 - torch.mean(center, axis=0) ** 2)
        self.score_fn = lambda z, t: torch.complie(torch.autograd.grad(self.logprior_density))(z, t)
        self.posterior_score_fn = lambda z, t, y: torch.complie(torch.autograd.grad(self.logposterior_density))(z, t, y)
        
    def mixture_prob(self, y):
        # posterior probability from a mixture component
        logwt = -torch.sum((self.center - y)**2, axis=-1) / 4
        wt = torch.exp(logwt - torch.max(logwt))
        prob = wt / torch.sum(wt)
        return prob
            
    def logprior_density(self, z, t):
        return torch.log(torch.exp(-torch.sum((z - self.center*torch.exp(-t))**2, axis=-1)/2).mean(axis=0))
    
    def sample_prior(self, shape):
        s = torch.randn(shape + (self.dim,))
        prob = torch.ones(self.n_center) / self.n_center
        idx = torch.multinomial(self.n_center, shape=shape, p=prob)
        s = s + self.center[idx]
        return s
    
    def logposterior_density(self, z, t, y):
        prob = self.mixture_prob(y)
        return torch.log(torch.sum(torch.exp(-torch.sum((z - (self.center+y)/2*torch.exp(-t))**2, axis=-1)/2/(1-torch.exp(-2*t)/2))*prob))
    
    def sample_posterior(self, shape, y):
        s = torch.randn(shape + (self.dim,)) * math.sqrt(1/2)
        prob = self.mixture_prob(y)
        dist = Categorical(probs=prob)
        idx = dist.sample((shape,)) 
        s = s + (self.center[idx] + y) / 2
        return s

def get_posterior(obs, prior, A, Sigma_y):
    modified_means = []
    modified_covars = []
    weights = []
    precision = torch.linalg.inv(Sigma_y)
    for loc, cov, weight in zip(prior.component_distribution.loc,
                                prior.component_distribution.covariance_matrix,
                                prior.mixture_distribution.probs):
        new_dist = gaussian_posterior(obs,
                                      A,
                                      torch.zeros_like(obs),
                                      precision,
                                      loc,
                                      cov)
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
        residue = obs - A @ new_dist.loc
        log_constant = -(residue[None, :] @ precision @ residue[:, None]) / 2 + \
                       prior_x.log_prob(new_dist.loc) - \
                       new_dist.log_prob(new_dist.loc)
        weights.append(torch.log(weight) + log_constant)
    # weights = torch.tensor(weights)
    # weights = weights[:, 0, 0]
    weights = torch.stack(weights)
    weights = weights[:, 0, 0]
    weights = weights - torch.logsumexp(weights, dim = 0)
    cat = Categorical(logits=weights)
    norm = MultivariateNormal(loc=torch.stack(modified_means, axis=0),
                                 covariance_matrix=torch.stack(modified_covars, axis=0))
    return MixtureSameFamily(cat, norm)

def gaussian_posterior(y,
                       likelihood_A,
                       likelihood_bias,
                       likelihood_precision,
                       prior_loc,
                       prior_covar):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (likelihood_A.T @ likelihood_precision @ (y - likelihood_bias) + prior_precision_matrix @ prior_loc)
    try:
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix)
    except ValueError:
        u, s, v = torch.linalg.svd(posterior_covariance_matrix, full_matrices=False)
        s = s.clip(1e-12, 1e6).real
        posterior_covariance_matrix = u.real @ torch.diag(s) @ v.real
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix)

class GaussMixTorch(Distribution):
    def __init__(self, center, weights=None, var_scale=1):
        super().__init__()
        if center.shape[1] is None:
            center.unsqueeze(0)
        self.dim = center.shape[1]
        self.center = center # of shape n_center * dim
        self.n_center = center.shape[0]
        if weights is None:
            weights = torch.ones(self.n_center) / self.n_center
        assert torch.abs(weights.sum()-1)<1e-6, "the sum of weights should be 1"
        self.var_scale = var_scale
        self.weights = weights
        self.A = torch.eye(self.dim)
        self.Sigma_y = torch.eye(self.dim)
        # assume the component variance is always 1
        covs = (torch.eye(self.dim).unsqueeze(0) * var_scale**2).repeat(self.n_center, 1, 1)
        self.prior_dist = MixtureSameFamily(
            Categorical(weights),
            MultivariateNormal(center, covs)
        )
        self.prior_logits = self.prior_dist.mixture_distribution.logits
        sigma0 = torch.sqrt(self.prior_dist.variance)
        # if not jnp.allclose(sigma0, jnp.ones_like(sigma0) * sigma0[0]):
        #     print("Warning: the covariance of the data distribution is not proportional to identity")
        self.sigma0 = sigma0[0]
        # self.score_fn = lambda z, t: jax.jit(grad(self.logprior_density))(z, t)
        def score_fn(z, t):
            z = z.clone().detach().requires_grad_(True)
            logp = self.logprior_density(z, t)
            grad_z = torch.autograd.grad(logp, z, grad_outputs=torch.ones_like(logp), create_graph=True)[0]
            return grad_z

        self.score_fn = score_fn
        # compute covariance matrix
        mean = self.prior_dist.mean
        comp_mean_centered = center - mean
        comp_cov = self.prior_dist.component_distribution.covariance_matrix + comp_mean_centered[:, :, None] * comp_mean_centered[:, None, :]
        self.Sigma_data = torch.sum(weights[:, None, None] * comp_cov, axis=0)
        assert torch.allclose(self.prior_dist.variance, torch.diag(self.Sigma_data)), "Inconsistency for convariance matrix"

    def set_observation(self, obs, A=None, Sigma_y=None):
        self.obs = obs
        if A is not None:
            self.A = A
        if Sigma_y is not None:
            self.Sigma_y = Sigma_y
        self.posterior_dist = get_posterior(obs, self.prior_dist, self.A, self.Sigma_y)
        tmp_mat = self.A.T @ self.A
        
        if torch.allclose(tmp_mat, torch.diag(torch.diag(tmp_mat)), atol=1e-5):
            def score_fn(z, t, y):
                z = z.clone().detach().requires_grad_(True)
                logp = self.logposterior_density(z, t, y, self.posterior_dist)
                grad_z = torch.autograd.grad(logp, z, grad_outputs=torch.ones_like(logp), create_graph=True)[0]
                return grad_z
            # self.posterior_score_fn = lambda z, t, y: jax.jit(grad(self.logposterior_density))(z, t, y, self.posterior_dist)
            self.posterior_score_fn = score_fn
        else:
            # self.posterior_score_fn = lambda z, t, y: jax.jit(grad(self.logposterior_density_general))(z, t, y, self.posterior_dist)
            def score_fn(z, t, y):
                z = z.clone().detach().requires_grad_(True)
                logp = self.logposterior_density_general(z, t, y, self.posterior_dist)
                grad_z = torch.autograd.grad(logp, z, grad_outputs=torch.ones_like(logp), create_graph=True)[0]
                return grad_z
            self.posterior_score_fn = score_fn
        # for pseudo-inverse computation
        self.sigma_y_var = self.Sigma_y[0][0] # always assume Sigma_y is proportional to identity
        # use the general covariance matrix of piror is not necessarily better than the diagonal part only
        # tmp_mat = jnp.linalg.inv(self.Sigma_data) + self.A.T @ self.A / self.sigma_y_var
        # when the convariance matrix of prior is porportional to identity
        tmp_mat = torch.eye(self.dim)/self.sigma0**2 + self.A.T @ self.A / self.sigma_y_var
        self.eigv, eigvec = torch.linalg.eigh(tmp_mat)
        self.composed_mat = self.A @ eigvec / self.sigma_y_var
        print("Obeservation model updated")

    def logprior_density(self, z, t):
        # up to a z-independent constant, assume the component variance is always self.var_scale**2 * identity
        var = self.var_scale ** 2 * torch.exp(-2*t) + 1 - torch.exp(-2*t)
        var = var.to(z.device)
        center = self.center.to(z.device)
        
        component_log_probs = -torch.sum((z.unsqueeze(1) - center.unsqueeze(0)*torch.exp(-t[0]))**2, dim=-1) / var.unsqueeze(1) / 2
        prior_logits = self.prior_logits.to(z.device)
        component_log_probs = F.log_softmax(prior_logits) + component_log_probs
        
        return torch.logsumexp(component_log_probs, dim=-1)
        # return jnp.log(jnp.sum(jnp.exp(-jnp.sum((z - self.center*jnp.exp(-t))**2, axis=-1)/2)*self.weights))

    def sample_prior(self, sample_shape):
        return self.prior_dist.sample(sample_shape)

    def sample(self, sample_shape):
        return self.sample_prior(sample_shape)

    def logposterior_density_general(self, z, t, y, posterior_dist):
        # for model where the posterior variance is general
        logits = posterior_dist.mixing_distribution.logits
        center = posterior_dist.component_mean
        cov = posterior_dist.component_distribution.covariance_matrix
        cov = cov*torch.exp(-2*t)+(1-torch.exp(-2*t))*torch.eye(self.dim)
        res = z - center*torch.exp(-t)
        component_log_probs = -torch.sum(res[:, None, :] @ torch.linalg.inv(cov) @ res[:, :, None], axis=(-2, -1)) / 2
        component_log_probs = F.log_softmax(logits) + component_log_probs
        return torch.logsumexp(component_log_probs, axis=-1)

    def logposterior_density(self, z, t, y, posterior_dist):
        # only for simple model where the posterior variance is all diagonal
        # prob = posterior_dist.mixing_distribution.probs
        logits = posterior_dist.mixing_distribution.logits
        center = posterior_dist.component_mean
        var = posterior_dist.component_variance
        # return jnp.log(jnp.sum(jnp.exp(-jnp.sum((z - center*jnp.exp(-t))**2/(var*jnp.exp(-2*t)+1-jnp.exp(-2*t)), axis=-1)/2)*prob))
        component_log_probs = -torch.sum((z - center*torch.exp(-t))**2/(var*torch.exp(-2*t)+1-torch.exp(-2*t)), dim=-1) / 2
        component_log_probs = F.log_softmax(logits) + component_log_probs
        return torch.logsumexp(component_log_probs, dim=-1)

    def sample_posterior(self, sample_shape):
        return self.posterior_dist.sample(sample_shape)
