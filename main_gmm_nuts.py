import os
import math
import jax
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import lax, vmap, jit
import json

from dist import GaussMixJax
from util import diffusion_sampler
from util import generate_inverse_problem_gm, sliced_wasserstein, analytical_boost_prior, forward_adapt
from numpyro.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from nuts import sample_nuts


@hydra.main(version_base=None, config_path="configs/", config_name="gmm")
def main(cfg):
    example_cfg = cfg.example
    key = jax.random.PRNGKey(cfg.random_key)
    n_groups = cfg.n_groups # each group is individual
    n_particles = example_cfg.n_particles
    n_steps = example_cfg.n_steps
    T = example_cfg.T
    dim = example_cfg.dim
    dim_y = example_cfg.dim_y
    kappa_range = example_cfg.kappa_range
    snr_range = example_cfg.snr_range
    noise = cfg.noise
    n_samples = cfg.n_samples
    dts = jnp.ones(n_steps) * T / n_steps

    key, subkey = random.split(key)
    center = []
    scale = 1
    for i in range(-2, 3): # jnp.array((-1,1)): 
        center += [jnp.array([-8.*scale*i, -8.*scale*j]*(dim//2)) for j in range(-2, 3)]  # jnp.array((-1,1))]
    
    # Use broadcasting to get (num_components, num_components, dim)
    diffs = jnp.array(center)[:, jnp.newaxis, :] - jnp.array(center)[jnp.newaxis, :, :]
    
    # Calculate Euclidean distance: sqrt(sum(diffs^2))
    dist_matrix = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
    max_mean_dist = jnp.max(dist_matrix)

    weights = random.uniform(subkey, (len(center),))**2
    weights = weights / weights.sum()
    target = GaussMixJax(jnp.array(center), weights, scale)

    # key, subkey = random.split(key)
    # z = random.normal(subkey, (n_groups, 1, dim))
    # tmp1 = jax.jit(jax.grad(target.logprior_density))(z, 0)
    # tmp2 = jax.jit(jax.grad(target.annealed_score_proxy))(z, 0)
    # print(jnp.linalg.norm(tmp1-tmp2))
    # exit()

    # ===== generate problem =====
    key, subkey = random.split(key)
    y, A, Sigma_y, x_origin = generate_inverse_problem_gm(subkey, dim, dim_y, target, scale, kappa_range, snr_range)
    var_y = Sigma_y[0][0]
    U, D, V = jnp.linalg.svd(A, full_matrices=False)
    V = V.T
    eps = 1e-2
    lambda_max = D.max()**2
    T_denoise = np.log(1+var_y/lambda_max)/2
    covs = jnp.repeat(jnp.eye(dim)[None]*(target.var_scale**2*jnp.exp(-(T_denoise-eps)*2)+1-jnp.exp(-(T_denoise-eps)*2)), axis=0, repeats=target.n_center)
    prior_middle = MixtureSameFamily(
        mixing_distribution=Categorical(target.weights),
        component_distribution=MultivariateNormal(jnp.array(center)*jnp.exp(-(T_denoise-eps)), covariance_matrix=covs)
    )
    A_singular = (jnp.exp(2*(T_denoise-eps)) / (1 + var_y/D**2 - jnp.exp(2*(T_denoise-eps))))**(1/2)
    A_singular = jnp.diag(A_singular) @ V.T
    aux_singular = (jnp.exp(2*(T_denoise-eps))/((1-jnp.exp(2*(T_denoise-eps)))*D**2+var_y))**(1/2)
    obs_singular = jnp.diag(aux_singular) @ (jnp.exp(-(T_denoise-eps)) * (U.T @ y))
    n_step_denoise = math.ceil(T_denoise / T * n_steps)
    dts_denoise = jnp.ones(n_step_denoise) * (T_denoise-eps) / n_step_denoise
    print(f"T denoise: {T_denoise}, n_step_denoise: {n_step_denoise}")
    print(f"var_y={var_y}")
    key, subkey = random.split(key)
    samples_true_full = target.sample_posterior(key, (n_samples,))
    plot_dim = 0
    nuts_steps = getattr(example_cfg, 'nuts_steps', 100)
    key, subkey = random.split(key)
    samples_prior = target.sample_prior(subkey, (n_groups, n_particles,))

    key, subkey = random.split(key)
    samples_boost_post_true = analytical_boost_prior(subkey,center, 0, weights, A, y, n_groups, n_particles, var_y)

    plt.figure()
    plt.scatter(samples_true_full[:,0], samples_true_full[:,1], color='g', label='Analytical', alpha=0.4, s=5)
    plt.scatter(samples_boost_post_true[:,0], samples_boost_post_true[:,1], color='b', label='Ref', alpha=0.4, s=5)
    plt.legend()
    plt.savefig("./fig/2d_test/analytical_posterior.png")
    plt.close()

    plt.figure()
    plt.scatter(samples_prior[0][:, 0], samples_prior[0][:, 1], color='r', label='Analytical Prior', alpha=0.4, s=5)
    plt.scatter(samples_true_full[:,0], samples_true_full[:,1], color='g', label='Ref Posterior', alpha=0.4, s=5)
    plt.scatter(samples_boost_post_true[:,0], samples_boost_post_true[:,1], color='b', label='Analytical Posterior', alpha=0.4, s=5)
    plt.legend()
    plt.savefig("./fig/2d_test/analytical_prior_post.png")
    plt.close()

    # ===== NUTS of posterior =====
    key, subkey = random.split(key)
    posterior_logprob_nuts = lambda x: - ((y - A @ x['loc']) ** 2).sum(axis=-1) / 2 / var_y + target.logprior_density(x['loc'], 0)
    samples_nuts, initial_positions_hmc = sample_nuts(dim, subkey, target, n_groups*n_particles, posterior_logprob_nuts, nuts_steps)
    
    sample1 = samples_true_full[:n_particles]
    sample2 = samples_nuts
    key, wass_subkey = random.split(key)
    swd_hmc_post = sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=1000)
    print("Sliced wasserstein distance of HMC: {}".format(swd_hmc_post))
    # print("Sliced wasserstein distance of HMC: {}".format(sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=100)))

    plt.figure()
    plt.scatter(samples_true_full[:,0], samples_true_full[:,1], color='r', label='Ref', alpha=0.4, s=5)
    plt.scatter(samples_nuts[:,0], samples_nuts[:,1], color='b', label='HMC', alpha=0.4, s=5)
    # plt.scatter(initial_positions_hmc[:,0], initial_positions_hmc[:,1], color='g', alpha=0.4, s=5)
    plt.legend()
    plt.savefig("./fig/2d_test/HMC_posterior.png")
    plt.close()

    # ===== NUTS of boosted posterior =====
    posterior_logprob_nuts = lambda x: - ((obs_singular - A_singular @ x['loc']) ** 2).sum(axis=-1) / 2 + target.logprior_density(x['loc'], T_denoise-eps)
    key, subkey = random.split(key)
    samples_nuts_init, initial_positions_boost = sample_nuts(dim, subkey, target, n_groups*n_particles, posterior_logprob_nuts, nuts_steps)

    # # check HMC initalizations
    # # key, wass_subkey = random.split(key)
    # # sample1 = initial_positions_boost
    # # sample2 = initial_positions_hmc
    # # swd_hmc_init = sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=1000)
    # # print("Sliced wasserstein distance of HMC inital postitions: {}".format(swd_hmc_init))
    # # exit()

    key, subkey = random.split(key)
    samples_boost_prior_true = analytical_boost_prior(subkey, center, T_denoise - eps, weights, A, y, n_groups, n_particles, var_y)

    key, wass_subkey = random.split(key)
    sample1 = samples_boost_prior_true
    sample2 = samples_nuts_init.reshape((-1, dim))
    swd_boost_prior = sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=1000)
    print("Sliced wasserstein distance of boosted prior: {}".format(swd_boost_prior))

    plt.figure()
    plt.scatter(samples_boost_prior_true[:,0], samples_boost_prior_true[:,1], color='r', label='Ref', alpha=0.4, s=5)
    plt.scatter(samples_nuts_init.reshape((-1, dim))[:,0], samples_nuts_init.reshape((-1, dim))[:,1], color='b', label='Tilted', alpha=0.4, s=5)
    # plt.scatter(initial_positions_boost[:,0], initial_positions_boost[:,1], color='g', label='Init', alpha=0.05, s=5, zorder=1)
    plt.legend(); plt.show()
    plt.savefig("./fig/2d_test/boost_prior.png")
    plt.close()

    cfg.reverse = True
    samples_nuts_init = samples_nuts_init.reshape((n_groups, n_particles, dim))
    key, subkey = random.split(key)
    keys = jax.random.split(subkey, n_groups)
    keys = jnp.array([jax.random.split(k, n_particles) for k in keys])
    samples_nuts_boost_full = diffusion_sampler(keys, samples_nuts_init, dts_denoise, cfg, target.score_fn)
    
    sample1 = samples_true_full[:n_particles]
    sample2 = samples_nuts_boost_full.reshape((-1, dim))
    swd_boost_post = sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=1000)
    print("Sliced wasserstein distance of boosted posterior: {}".format(swd_boost_post))

    plt.figure()
    plt.scatter(samples_true_full[:,0], samples_true_full[:,1], color='r', label='Ref', alpha=0.4, s=5)
    plt.scatter(samples_nuts_boost_full.reshape((-1, dim))[:,0], samples_nuts_boost_full.reshape((-1, dim))[:,1], color='b', label='Tilted', alpha=0.4, s=5)
    plt.legend()
    plt.savefig("./fig/2d_test/boosted_posterior.png")
    plt.close()

    # ===== Iterated boosted posterior =====
    # Initialization via annealed Langevin
    key, subkey = random.split(key)
    keys = jax.random.split(subkey, n_groups) # keys.shape (2,2)
    keys = jnp.array([jax.random.split(k, n_particles) for k in keys])
    x0s = random.normal(subkey, (n_groups, n_particles, dim))  
    sigma_max = max_mean_dist # / dim
    sigma_min = 0.01
    # dim: 20, dim_y: 1
    noise_steps = 200
    ald_steps = 50
    # dim: 40, dim_y: 1
    # noise_steps = 200
    # ald_steps = 50
    # dim: 80, dim_y: 1
    # noise_steps = 300
    # ald_steps = 50

    # A global Python set to remember which sigmas we've already printed
    # _printed_sigmas = set()

    # def _print_interceptor(sigma, p_norm, l_norm):
    #     # Round the sigma value slightly to group identical steps together
    #     s_val = round(float(sigma), 5)
        
    #     # Only print if we haven't seen this sigma value yet during this run
    #     if s_val not in _printed_sigmas:
    #         print(f"Sigma: {s_val:.4f} | Prior Norm: {float(p_norm):.4f} | Likelihood Norm: {float(l_norm):.4f}")
    #         # Add it to the set so the other 999 particles are ignored
    #         _printed_sigmas.add(s_val)

    # vanilla score
    def total_score_fn(z, sigma):
        prior_score = jax.grad(target.annealed_score_proxy)(z, sigma)
        likelihood_score = -A.T @ (A @ z - y) / var_y
        
        return prior_score + likelihood_score

    # DPS score
    def dps_score_fn(z, sigma): #, A, y, var_y):
        # 1. The prior score (exactly as you had it)
        prior_score = jax.grad(target.annealed_score_proxy)(z, sigma)
        
        # 2. Define the likelihood using Tweedie's estimate
        def log_likelihood_fn(z_val):
            # Recompute prior score inside to allow jax.grad to track the computation graph
            p_score = jax.grad(target.annealed_score_proxy)(z_val, sigma)
            
            # Tweedie's estimate of the clean state
            x_hat = z_val + (sigma**2) * p_score
            
            # Log-likelihood of the estimated clean state
            # Note: We return the scalar log-density, not the gradient
            # return -0.5 * jnp.sum(jnp.square(A @ x_hat - y)) / var_y
            return -1. * jnp.sum(jnp.square(A @ x_hat - y))

        # 3. Get the likelihood score by differentiating through Tweedie's estimate
        # likelihood_score = jax.grad(log_likelihood_fn)(z) 
        likelihood_score = jax.grad(log_likelihood_fn)(z) / jnp.sqrt(-log_likelihood_fn(z)) * 1.2

        # 4. Conditional Print
        # p_norm = jnp.linalg.norm(prior_score)
        # l_norm = jnp.linalg.norm(likelihood_score)
        # jax.debug.callback(_print_interceptor, sigma, p_norm, l_norm)
        
        # 5. Combine for the true annealed posterior score
        return prior_score + likelihood_score
    
    def annealed_langevin_fast(key, x0, score_fn, sigma_max, sigma_min, L, T, alpha=2e-5):
        sigmas = jnp.geomspace(sigma_max, sigma_min, num=L)
        etas = alpha * (sigmas / sigmas[-1])**2

        # Inner scan: T steps at a fixed sigma
        def inner_step(carry, _):
            x, key = carry
            key, subkey = random.split(key) # k must be shape (2,)
            score = score_fn(x, current_sigma_val)
            
            # Langevin update
            next_x = x + current_eta_val * score + jnp.sqrt(2 * current_eta_val) * random.normal(subkey, x.shape)
            return (next_x, key), None

        # Outer scan: L noise levels
        def outer_step(carry, i):
            x, k = carry
            # We need to make these available to the inner function
            # Using non-local variables in a nested scan works in JAX
            nonlocal current_sigma_val, current_eta_val
            current_sigma_val = sigmas[i]
            current_eta_val = etas[i]

            # jax.debug.print("Processing Sigma Squared Step {i}: sigma_squared={s}", i=i, s=current_sigma_val**2)
            # jax.debug.print("Processing Eta Step {i}: eta={s}", i=i, s=current_eta_val)

            (x_next, k_next), _ = jax.lax.scan(inner_step, (x, k), jnp.arange(T))
            return (x_next, k_next), None

        # Initial current values for the closure
        current_sigma_val = sigmas[0]
        current_eta_val = etas[0]

        (final_x, _), _ = jax.lax.scan(outer_step, (x0, key), jnp.arange(L))
        return final_x

    # We want to map over 'key' and 'x0'
    # in_axes: (key=0, x0=0, score_fn=None, sigma_max=None, ...)
    # None means that argument is shared across all parallel runs.
    vmapped_ald = vmap(annealed_langevin_fast, in_axes=(0, 0, None, None, None, None, None))

    # 3. For the (n_groups, n_particles) structure, we vmap twice
    # This handles the nested batch dimensions effectively
    parallel_ald = vmap(vmapped_ald, in_axes=(0, 0, None, None, None, None, None))

    # 4. JIT the entire batch operation for maximum speed
    final_sampler = jit(parallel_ald, static_argnums=(2, 5, 6))

    # Split into (n_groups, n_particles, 2)
    keys = jax.random.split(key, n_groups * n_particles).reshape(n_groups, n_particles, 2)

    # Create a tiny version of your inputs
    # Wrap a simple jax.debug.print inside your score_fn temporarily
    def debug_score_fn(z, sigma):
        score = total_score_fn(z, sigma)
        # This will print for EVERY particle passed to this specific call
        jax.debug.print("Score norm: {s}", s=jnp.linalg.norm(score))
        return score
    
    # Run just one particle through a non-vmapped version
    # debug_result = annealed_langevin_fast(
    #     keys[0,0], x0s[0,0], debug_score_fn, sigma_max, sigma_min, noise_steps, ald_steps
    # )

    # exit()

    # Run the sampler
    # x0s shape: (n_groups, n_particles, dim)
    final_samples = final_sampler(
        keys, 
        x0s, 
        dps_score_fn, 
        float(sigma_max), 
        float(sigma_min), 
        int(noise_steps), 
        int(ald_steps)
    )

    key, subkey = random.split(key)
    n_fold = 5
    random_integers = jax.random.randint(key, shape=(int(n_particles/n_fold),), minval=0, maxval=1000)
    final_samples = jnp.repeat(final_samples[:,random_integers,:], repeats=n_fold, axis=1)

    sample1 = samples_true_full
    sample2 = final_samples.reshape((-1, dim))
    key, wass_subkey = random.split(key)
    swd_annealed = sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=1000)
    print("Sliced wasserstein distance of annealed LMC: {}".format(swd_annealed))

    plt.figure()
    plt.scatter(samples_true_full[:,0], samples_true_full[:,1], color='r', label='Ref', alpha=0.4, s=5)
    plt.scatter(final_samples.reshape((-1, dim))[:,0], final_samples.reshape((-1, dim))[:,1], color='b', label='Annealed', alpha=0.4, s=5)
    plt.legend()
    plt.savefig("./fig/2d_test/annealed_initialization.png")
    plt.close()

    # exit()

    # Iterated sampling
    cfg.reverse = True
    # x_init = jnp.linalg.pinv(A) @ y + random.normal(subkey, (n_groups, n_particles, dim)) # * 0.1 
    # x_init = jnp.mean(samples_true_full, axis=0) + random.normal(subkey, (n_groups, n_particles, dim)) * 0.1
    # x_init = random.normal(subkey, (n_groups, n_particles, dim)) * 30
    # x_init = samples_nuts.reshape((n_groups, n_particles, dim))
    x_init = final_samples
    samples_iter = x_init + random.normal(subkey, (n_groups, n_particles, dim)) * 1.0 

    K = 100
    betas = jnp.ones(K) * 1.0 
    # betas = jnp.concatenate([jnp.full((int(K / 10 * 4),), 5), jnp.full((int(K / 10 * 3),), 3), jnp.full((int(K / 10 * 3),), 1)])
    N_save = 1
    print(f"K={K}")
    # print(f"beta={betas}")
    swd_forward = []
    swd_reverse = []

    n_step_denoise = math.ceil(T_denoise / T * n_steps)
    dts_denoise = jnp.ones(n_step_denoise) * (T_denoise-eps) / n_step_denoise
    dt_denoise = dts_denoise[0]

    # key, subkey = random.split(key)
    # samples_boost_t_true = analytical_boost_prior(subkey, center, T_denoise - eps, weights, A, y, n_groups, n_particles, var_y)
    key, subkey = random.split(key)
    keys = jax.random.split(subkey, n_groups)
    keys = jnp.array([jax.random.split(k, n_particles) for k in keys])
    samples_test = diffusion_sampler(keys, jnp.expand_dims(samples_boost_prior_true, axis=0), dts_denoise, cfg, target.score_fn)

    sample1 = samples_true_full#samples_boost_prior_true# .reshape((-1, dim))
    sample2 = samples_test.reshape((-1, dim))
    key, wass_subkey = random.split(key)
    swd_reverse_gt = sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=100)
    print(f"Sliced wasserstein distance of test: {swd_reverse_gt}")

    samples_boost_t_true = analytical_boost_prior(subkey, center, dt_denoise, weights, A, y, n_groups, n_particles, var_y)

    for k in range(K):
        print(f"iteration {k+1}")
        # cfg.reverse = True
        key, subkey = random.split(key)
        keys = jax.random.split(subkey, n_groups)
        keys = jnp.array([jax.random.split(k, n_particles) for k in keys])
        # samples_iter, ts = forward_adapt(key, keys, samples_iter, dt_denoise, 0, (T_denoise-eps), betas[k], A, y, var_y) #, center, weights)
        samples_iter, ts = forward_adapt(key, keys, samples_iter, dt_denoise, 0, dt_denoise+eps**3, betas[k], A, y, var_y) # , center, weights)
        samples_boost_t_true = analytical_boost_prior(subkey, center, ts[-1], weights, A, y, n_groups, n_particles, var_y)

        # Metropolis adjustment

        
        key, subkey = random.split(key)
        sample1 = samples_boost_t_true
        sample2 = samples_iter.reshape((-1, dim))
        key, wass_subkey = random.split(key)
        swd_forward.append(sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=100))
        # swd_forward.append(swd[-1])
        print(f"iteration {k+1}")
        print(f"Sliced wasserstein distance of boosted prior iter {k+1}: {swd_forward[-1]}")

        key, subkey = random.split(key)
        keys = jax.random.split(subkey, n_groups)
        keys = jnp.array([jax.random.split(k, n_particles) for k in keys])
        n_step_adapt = math.ceil(ts[-1] / T * n_steps)
        dts_adapt = jnp.ones(n_step_adapt) * ts[-1] / n_step_adapt
        # samples_iter = diffusion_sampler(keys, samples_iter, dts_adapt, cfg, target.score_fn) # for T >> dt
        samples_iter = diffusion_sampler(keys, samples_iter, jnp.array([dts_adapt[0]]), cfg, target.score_fn) # for T = dt

        sample1 = samples_true_full # .reshape((-1, dim))
        sample2 = samples_iter.reshape((-1, dim))
        key, wass_subkey = random.split(key)
        swd_reverse.append(sliced_wasserstein(wass_subkey, sample1, sample2, n_slices=100))
        print(f"iteration {k+1}")
        print(f"Sliced wasserstein distance of posterior iter {k+1}: {swd_reverse[-1]}")
            
    plt.figure()
    plt.scatter(samples_true_full[:,0], samples_true_full[:,1], color='r', label='Ref', alpha=0.4, s=5)
    plt.scatter(samples_iter.reshape((-1, dim))[:,0], samples_iter.reshape((-1, dim))[:,1], color='b', label='Iter', alpha=0.4, s=5)
    plt.legend()
    plt.savefig("./fig/2d_test/iter_posterior.png")
    plt.close()

    plt.figure()
    plt.plot(jnp.arange(1,K+1), jnp.array(swd_forward), color='b',  label=f'forward={swd_forward[-1]:.4f}', marker='o')
    plt.plot(jnp.arange(1,K+1), jnp.array(swd_reverse), color='orange', label=f'reverse={swd_reverse[-1]:.4f}', marker='x')
    # plt.axhline(y=swd_reverse_gt, color='black', linestyle='--',  label=f'reverse_gt={swd_reverse_gt:.4f}')
    plt.axhline(y=swd_boost_prior, color='blue', linestyle='--', label=f'boost_prior={swd_boost_prior:.4f}')
    plt.axhline(y=swd_boost_post, color='orange', linestyle='--', label=f'boost_post={swd_boost_post:.4f}')
    plt.axhline(y=swd_hmc_post, color='g', linestyle='--', label=f'hmc_post={swd_hmc_post:.4f}')
    plt.axhline(y=swd_annealed, color='r', linestyle='--', label=f'ald_post={swd_annealed:.4f}')
    plt.plot()
    plt.xlabel('iterations')
    plt.ylabel('SWD')
    plt.legend()
    plt.show()
    plt.savefig("./fig/2d_test/swd.png")
    plt.close()

    log_file = './exp_log_nuts_dt0.05.jsonl'

    def log_result_jsonl(log_file, result_dict):
        with open(log_file, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')

    result = {
        'exp_seed': cfg.random_key,
        'data_dim': int(dim),
        'obs_dim': int(dim_y),
        'noise var': float(var_y),
        'hmc': float(swd_hmc_post),
        'boost': float(swd_boost_post),
        'iterative': float(swd_reverse[-1])
    }
    log_result_jsonl(log_file, result)



if __name__ == "__main__":
    main()
    os._exit(0)