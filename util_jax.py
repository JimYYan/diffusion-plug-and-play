import time
import jax
import jax.numpy as jnp
from jax import random
from jax import lax, vmap, jit
from functools import partial
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

group_resample = vmap(lambda param, prob, key: random.choice(key, param, shape=(param.shape[0],), p=prob))

# adaptived from https://www.jeremiecoullon.com/2020/11/10/mcmcjax3ways/
# @partial(jit, static_argnums=(3,4))
@partial(jit, static_argnums=(3,))
def transition_kernel_sde(key, param, t, drift_fn, dt):
    # original
    # key, subkey = random.split(key)
    # drift = drift_fn(param, t)
    # # score = (drift - param) / 2
    # param = param - dt*(-drift) - jnp.sqrt(2*dt)*random.normal(key=subkey, shape=(param.shape))
    # return key, param

    # diffpnp diffusion kernel
    key, subkey = random.split(key)
    drift = drift_fn(param, t)
    score = (drift - param) / 2
    param = param - dt*(-score) - jnp.sqrt(2*dt)*random.normal(key=subkey, shape=(param.shape))
    return key, param

@partial(jit, static_argnums=(3,4))
def transition_kernel_ode(key, param, t, drift_fn, dt):
    drift = drift_fn(param, t)
    param = param - dt*(-drift)
    return key, param

def diffusion_sampler(keys, x0s, dts, cfg, score_fn):
    start_time = time.monotonic()
    results = diffusion_sampler_jax(keys, x0s, dts, cfg, score_fn)
    # print("Elapsed time for sampling: {:.1f}s".format(time.monotonic() - start_time))
    return results

@partial(jit, static_argnums=(3,4))
# @partial(jit, static_argnums=(3,))
def diffusion_sampler_jax(keys, x0s, dts, cfg, score_fn):
    # keys: shape of [n_groups, n_particles, 2]
    # x0s: shape of [n_groups, n_particles, dim]
    # dts: shape of [n_steps], ts = cumsum(dts), the last one is the first step of reverse process
    reverse = cfg.reverse
    noise = cfg.noise
    n_groups = x0s.shape[0]
    n_steps = dts.shape[0]
    ts = jnp.cumsum(dts)
    idxs = jnp.arange(n_steps+1, 1, -1)
    fixed_arrs = jnp.stack([idxs, ts, dts], axis=1)
    fixed_arrs = jnp.flip(fixed_arrs, axis=0)
    if reverse:
        pass
    else: # Langevin
        noise = True
        ts = ts * 0
    if reverse:
        if noise:
            drift_fn = lambda z,t: 2*score_fn(z, t) + z
        else: # reverse ODE
            drift_fn = lambda z,t: score_fn(z, t) + z
    else: # Langevin
        drift_fn = lambda z,t: score_fn(z, t)
    def particle_step(i, carry):
        keys, params = carry
        idx, t, dt = fixed_arrs[i]
        idx = idx.astype(int)
        if noise:
            # keys_next, params_next = vmap(vmap(lambda key, param: transition_kernel_sde(key, param, t, drift_fn, dt)))(keys, params)
            
            # Define the inner mapping (over params)
            # in_axes: key (0), param (0), t (None), drift_fn (None), dt (None)
            inner_vmap = vmap(
                lambda key, param: transition_kernel_sde(key, param, t, drift_fn, dt), 
                in_axes=(0, 0)
            )
            # Define the outer mapping (over keys)
            outer_vmap = vmap(inner_vmap, in_axes=(0, 0))
            keys_next, params_next = outer_vmap(keys, params)
        else:
            keys_next, params_next = vmap(vmap(lambda key, param: transition_kernel_ode(key, param, t, drift_fn, dt)))(keys, params)
        # jax.debug.print("Step {}, value={}", i, jnp.linalg.norm(((drift - params_next) / 2)[0][0]))
        return keys_next, params_next#, score
    carry = (keys, x0s)
    _, samples = lax.fori_loop(0, n_steps, particle_step, carry)
    # scores = []
    # sample = []
    # for i in range(n_steps):
    #     carry = particle_step(i, carry)
    #     # scores.append(score)
    #     # sample.append(samples)
    #     # print(jnp.array(sample[-1][:10]))
    # samples = carry[1]
    # print(samples.shape)
    return samples # , scores

def generate_inverse_problem_gm(key, dim, dim_y, target, scale=1, kappa_range=None, snr_range=None):
    # # Generate A for low-dim experiments
    # # Step 1: Random full-rank matrix
    # A = jax.random.normal(key, (dim, dim))

    # # Step 2: QR decomposition to get orthonormal rows
    # Q, R = jnp.linalg.qr(A)  # Q: 20x20, rows are orthonormal

    # # Step 3: Apply anisotropic scaling across rows
    # row_scales = jnp.geomspace(1.0, 0.1, dim)  # decay from 1 to 0.01
    # A = Q * row_scales[:, None]  # scale each row

    # A = A[:dim_y, :]

    A = random.normal(key, (dim_y, dim))
    U, D, V = jnp.linalg.svd(A, full_matrices=True)
    
    coordinate_mask = np.ones_like(V[0])
    coordinate_mask[len(D):] = 0
    key, subkey = random.split(key)
    if kappa_range is not None or snr_range is not None:
        assert dim_y == dim, "Only support non-degegenerate observation for setting kappa or SNR; otherwise, by definition kappa = infty and SNR=0 since lambda_min=0"
    if kappa_range is None:
        diag = jnp.sort(random.uniform(subkey, D.shape) * -1) * -1

    else:
        kappa = random.uniform(subkey, (1,), minval=kappa_range[0], maxval=kappa_range[1])[0]
        diag = jnp.sort(random.uniform(subkey, D.shape, minval=1/kappa, maxval=1.0) * -1) * -1
        diag = diag.at[0].set(1)
        diag = diag.at[-1].set(1/kappa)
    A = U @ (jnp.diag(diag) @ V[coordinate_mask == 1, :]) / scale
    # A = jnp.concatenate([U, jnp.zeros((dim_y, dim - dim_y))], axis=1) # for orthogonal A
    # A = jnp.concatenate([jnp.eye(dim_y), jnp.zeros((dim_y, dim - dim_y))], axis=1)
    key, subkey = random.split(key)
    init_sample = target.prior_dist.sample(subkey, sample_shape=())
    key, subkey = random.split(key)
    if snr_range is None:
        # var_observations = random.uniform(subkey, (1,), minval=0.05, maxval=0.050001)[0] * jnp.ones(len(diag))
        # var_observations = random.uniform(subkey, (1,), minval=0.5, maxval=0.50001)[0] * jnp.ones(len(diag))
        var_observations = random.uniform(subkey, (1,), minval=1.0, maxval=1.01)[0] * jnp.ones(len(diag))
        # var_observations *= 4
    else:
        var_observations = jnp.ones(len(diag)) * min(diag)**2 / random.uniform(subkey, (1,), minval=snr_range[0], maxval=snr_range[1])[0]
    std = var_observations**0.5
    # var_observations = std**2
    if dim == 1:
        init_obs = A * init_sample
    else:
        init_obs = A @ init_sample
    key, subkey = random.split(key)
    init_obs += random.normal(subkey, init_obs.shape) * (std)
    Sigma_y = jnp.diag(var_observations)
    target.set_observation(init_obs, A, Sigma_y)
    return init_obs, A, Sigma_y, init_sample

def sliced_wasserstein(key, sample1, sample2, n_slices=100):
    projections = random.normal(key, shape=(n_slices, sample1.shape[1]))
    projections = projections / jnp.linalg.norm(projections, axis=-1, keepdims=True)
    sample1_projected = (projections @ sample1.T)
    sample2_projected = (projections @ sample2.T)
    return np.mean([wasserstein_distance(u_values=s1, v_values=s2) for s1, s2 in zip(sample1_projected, sample2_projected)])

def save_fig_slice(samples_full, plot_dim, samples_true_full, x0s, file_name):
    samples_langevin = samples_full[..., plot_dim].reshape([-1])
    samples_true = samples_true_full[:, plot_dim]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    cmap = plt.get_cmap("tab10")
    sns.kdeplot(samples_langevin, color='g', ax=ax, label='Samples')
    sns.kdeplot(samples_true, color='r', ax=ax, label='Ground Truth')
    if x0s is not None:
        sns.kdeplot(x0s[..., plot_dim].reshape([-1]), color='b', ax=ax, label='Ref')
    plt.legend(); plt.show()

    plt.savefig(f"./fig/slice/{file_name}.png")
    plt.close()
    return samples_langevin, samples_true

def save_fig_2d_scatter(samples_full, samples_true_full, x0s, file_name):
    samples_langevin = samples_full.reshape(-1, samples_full.shape[2])
    samples_true = samples_true_full

    plt.figure()
    plt.scatter(samples_true[:,0], samples_true[:,1], color='r', label='Ground Truth', alpha=0.4, s=5)
    if x0s is not None:
        plt.scatter(x0s.reshape(-1, x0s.shape[2])[:,0], x0s.reshape(-1, x0s.shape[2])[:,1], color="b",label='Ref', alpha=0.4, s=5)
    plt.scatter(samples_langevin[:,0], samples_langevin[:,1], color='g', label='Samples', alpha=0.4, s=5)
    plt.legend(); plt.show()

    plt.savefig(f"./fig/2d/{file_name}.png")
    plt.close()
    return samples_langevin, samples_true

def save_fig_1d(samples_full, samples_true_full, x0s, file_name):
    return 

def multivariate_gaussian_logpdf(x, mu, cov):
    # d = x.shape[-1]
    # diff = x - mu
    # chol = jnp.linalg.cholesky(cov)  # Lower triangular
    # log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    # inv_cov_diff = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, diff))
    # # quad_form = jnp.dot(diff, inv_cov_diff)
    # quad_form = jnp.sum(diff * inv_cov_diff, axis=-1)
    # norm_const = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det)
    # return jnp.exp(norm_const - 0.5 * quad_form)

    d = x.shape[-1]
    diff = x - mu
    chol = jnp.linalg.cholesky(cov)
    
    # Stable Log-Determinant
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    
    # Solve for quadratic form
    inv_cov_diff = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, diff.T)).T
    quad_form = jnp.sum(diff * inv_cov_diff, axis=-1)
    
    log_prob = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det + quad_form)
    return log_prob
    
def analytical_boost_prior(key, center, t, weights, meas, obs, n_groups, n_particles, var_y):
    dim = meas.shape[1]
    dim_y = meas.shape[0]
    U, D, V = jnp.linalg.svd(meas, full_matrices=False)
    V = V.T

    A_singular = A_t(t, D, V, var_y)
    obs_singular = y_tilde(t, U, D, obs, var_y)

    center_prior = jnp.exp(-t) * jnp.array(center)
    cov_prior = jnp.eye(dim) #covs[0]

    cov_post = jnp.linalg.inv(jnp.linalg.inv(cov_prior) + A_singular.T @ A_singular)
    cov_post = jnp.tile(cov_post, (jnp.array(center).shape[0],1,1))

    center_post = vmap(lambda v: jnp.linalg.inv(cov_prior) @ v + A_singular.T @ obs_singular)(center_prior) 
    center_post = vmap(jnp.matmul, in_axes=(0, 0))(cov_post, center_post)
    cov_prior = jnp.tile(cov_prior, (jnp.array(center).shape[0],1,1))
    log_likelihoods = vmap(lambda mu, cov: multivariate_gaussian_logpdf(obs_singular, A_singular @ mu, A_singular @ cov @ A_singular.T + jnp.eye(dim_y)))(center_prior, cov_prior)
    log_weights_post = jnp.log(weights) + log_likelihoods
    log_total_evidence = jax.nn.logsumexp(log_weights_post)
    weight_post = jnp.exp(log_weights_post - log_total_evidence)
    # weight_post = weights * weight_post
    weight_post = weight_post / (jnp.sum(weight_post))

    key, subkey = random.split(key)
    component_indices = random.choice(subkey, a=weight_post.shape[0], shape=(n_groups * n_particles,), p=weight_post)
    key, subkey = random.split(key)
    z = random.normal(subkey, shape=(n_groups * n_particles, dim))  
    mu_samples = center_post[component_indices]               
    cov_samples = cov_post[component_indices]             
    L = jnp.linalg.cholesky(cov_samples)            
    samples_boost_prior_true = mu_samples + jnp.einsum('nij,nj->ni', L, z) 
    return samples_boost_prior_true

def A_t(t, D, V, var_y):
    A_singular = (jnp.exp(2*t) / (1 + var_y/D**2 - jnp.exp(2*t)))**(1/2)
    A_singular = jnp.diag(A_singular) @ V.T 

    return A_singular

def y_tilde(t, U, D, y, var_y):
    aux_singular = (jnp.exp(2*t)/((1-jnp.exp(2*t))*D**2+var_y))**(1/2)
    obs_singular = jnp.diag(aux_singular) @ (jnp.exp(-t) * (U.T @ y)) 
    
    return obs_singular

def N_step_langevin(key, x, t, dt, score_fn, n, var_y, meas, y, eps=1):
    U, D, V = jnp.linalg.svd(meas, full_matrices=False)
    V = V.T
    A_singular = A_t(t, D, V, var_y)
    obs_singular = y_tilde(t, U, D, y, var_y)
    for i in range(n):
        key, subkey = random.split(key)
        posterior_score = score_fn(x, t) - A_singular.T @ (A_singular @ x - obs_singular) * 1
        x = x - posterior_score * dt + jnp.sqrt(2*dt*eps)*random.normal(key=subkey, shape=(x.shape))
    
    return key, x

def swd_eval(key, center, t, weights, meas, obs, samples_recon):
    # dim = meas.shape[1]
    # dim_y = meas.shape[0]
    key, subkey = random.split(key)
    sample1 = analytical_boost_prior(subkey, center, t, weights, meas, obs)
    sample2 = samples_recon#.reshape((-1, dim))
    key, wass_subkey = random.split(key)

    return sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=1000)

def Lt_grad(x, t, meas, obs, var_y):
    U, D, V = jnp.linalg.svd(meas, full_matrices=False)
    V = V.T
    A_singular = A_t(t, D, V, var_y)
    obs_singular = y_tilde(t, U, D, obs, var_y)

    return -A_singular.T @ A_singular @ x + A_singular.T @ obs_singular

def forward_onestep(x, t, dt, key, meas, obs, var_y):
    key, subkey = random.split(key)
    grad_Lt = Lt_grad(x, t, meas, obs, var_y)
    x = x - x * dt + jnp.sqrt(2*dt)*random.normal(key=subkey, shape=(x.shape)) + 2 * grad_Lt * dt 
    
    return key, x

def forward(x, dts, keys, meas, obs):
    n_steps = dts.shape[0]
    ts = jnp.cumsum(dts)
    idxs = jnp.arange(n_steps+1, 1, -1)
    fixed_arrs = jnp.stack([idxs, ts, dts], axis=1)
    
    params = x
    for i in range(n_steps):
        idx, t, dt = fixed_arrs[i]
        idx = idx.astype(int)
    
        keys, params = vmap(vmap(lambda key, param: forward_onestep(param, t, dt, key, meas, obs)))(keys, params)

    return params 

def forward_adapt(key, keys, params, dt, t_start, T, beta, meas, obs, var_y): #, center, weights):
    # dim = meas.shape[1]
    # dim_y = meas.shape[0]
    ts = []
    # ts.append(t_start)
    t = t_start
    # swd = []
    # Lt_norm = []
    
    grad_Lt_norms = vmap(lambda param: jnp.linalg.norm(Lt_grad(param, t, meas, obs, var_y)))(params[0])
    dt_adapt = beta * dt / jnp.max(grad_Lt_norms) # used for T >> dt
    dt_adapt = dt # used for T = dt

    while t + dt_adapt < T:
        keys, params = vmap(vmap(lambda key, param: forward_onestep(param, t, dt_adapt, key, meas, obs, var_y)))(keys, params)
        t += dt_adapt
        ts.append(t)
        # key, subkey = random.split(key)
        # swd.append(swd_eval(subkey, center, t, weights, meas, obs, params.reshape(-1, dim)))
        grad_Lt_norms = vmap(lambda param: jnp.linalg.norm(Lt_grad(param, t, meas, obs, var_y)))(params[0])
        norm_max = jnp.max(grad_Lt_norms)
        dt_adapt = beta * dt / norm_max
        # dt_adapt = dt
        # Lt_norm.append(norm_max)
    return params, ts #, swd , Lt_norm
    