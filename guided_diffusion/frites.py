"""
FRITES sampler for image inverse problems.

Exact PyTorch port of the iterative algorithm (lines 342-414, main_gmm_nuts.py),
with the forward SDE re-parameterised to use the DDPM beta schedule.

Jax used an OU forward SDE:  dx_t = -x_t dt + sqrt(2) dW_t
  => q(x_t | x_0) = N(exp(-t)*x_0, (1-exp(-2t))*I),  alpha_bar_t = exp(-2t)

The DDPM reverse process uses a different noise schedule:
  q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
  where alpha_bar_t = prod(1-beta_i) from the DDPM beta schedule.

Both have the same structural form, but different alpha_bar values for the same sigma.  To keep forward and reverse consistent, this file uses the DDPM's actual alpha_bar (from self.alphas_cumprod) everywhere, found with the same time_mask logic used inside DDPMDenoiser.gen_schedule:
  time_mask      = alphas_cumprod > 1 / (1 + sigma^2)
  alpha_bar_ddpm = alphas_cumprod[time_mask][-1]   # last value still above threshold

Forward step (VP-SDE, big step from t=0 to t corresponding to alpha_bar_ddpm):
  grad_Lt      = A^T(y - A(x)) / var_y = -sq_norm_grad / (2*var_y)
  x_corrected  = x + (1-alpha_bar)*grad_Lt/var_y ... simplified as:
               = x - ((1-alpha_bar) / (2*sigma_y^2)) * sq_norm_grad
  x_noisy      = sqrt(alpha_bar)*x_before + (x_corrected - x_before) + sqrt(1-alpha_bar)*eps
               = sqrt(alpha_bar)*x_before + (1-alpha_bar)*grad_Lt   + sqrt(1-alpha_bar)*eps

  The gradient coefficient (1-alpha_bar) replaces the OU coefficient 2*dt = sigma^2.
  For small sigma they are numerically close; for larger sigma the DDPM value is exact.

DDS reverse step:  x = DDPMDenoiser.denoise(x_noisy, sigma, model)

Usage
-----
Add one import line to sample_condition.py before create_sampler is called:
    import guided_diffusion.pcs_dds   # registers 'pcs-dds' and 'forward_sde'

Diffusion config:  sampler: pcs-dds
                   params:  {eta_max, eta_min, k, k_start}

Task config:       conditioning:
                     method: forward_sde
                     params:
                       sigma_y: 0.05    # measurement noise std sqrt(var_y)
"""

import datetime
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm.auto import tqdm

from .gaussian_diffusion import (
    SpacedDiffusion,
    DDPMDenoiser,
    register_sampler,
    create_sampler,
)
from .condition_methods import ConditioningMethod, register_conditioning_method
from util.img_utils import clear_color, adjust_color, prepare_im
import lpips
from skimage.metrics import peak_signal_noise_ratio


# ---------------------------------------------------------------------------
# Forward conditioning: gradient part of the Jax forward_onestep
# ---------------------------------------------------------------------------

@register_conditioning_method(name='forward_sde')
class ForwardSDECondition(ConditioningMethod):
    """
    Gradient part of the Jax forward_onestep, with NO noise (noise is injected in the sampler after this call).

    Jax:  x_corrected = x + 2*dt * grad_Lt(x)
                      = x - (dt / var_y) * sq_norm_grad
    where dt = gamma^2 / 2 and sq_norm_grad = grad(||y - A(x)||^2).

    Config params
    -------------
    sigma_y : measurement noise std, i.e. sqrt(var_y).  Default 0.05.
    """

    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.sigma_y = kwargs.get('sigma_y', 0.1)
        self.N_gd = kwargs.get('N_gd', 1)
        self.lr = kwargs.get('lr', 1.0)

    def conditioning(self, x_t, measurement, **kwargs):
        gamma     = kwargs.get('gamma', self.sigma_y)
        alpha_bar = kwargs.get('alpha_bar', 1.0 / (1.0 + float(gamma) ** 2))

        # x_0_hat is not needed; gradient is computed directly at x_t.
        # x_prev == x_t == x_0_hat in the tilted SDE setting, so autograd
        # correctly gives grad_{x_t} ||y - A(x_t)||^2.
        # sq_norm_grad, sq_norm = self.sq_grad_and_value(
        #     x_prev=x_prev, x_0_hat=x_prev,
        #     measurement=measurement,
        #     **{k: v for k, v in kwargs.items() if k not in ('x_prev', 'x_0_hat', 'measurement')},
        # )
        # difference = measurement - self.operator.forward(x_t, **kwargs)
        # sq_norm = torch.linalg.norm(difference) ** 2
        # sq_norm_grad = torch.autograd.grad(outputs=sq_norm, inputs=x_t)[0]

        # Tilted forward SDE: gradient coefficient = (1 - alpha_bar) / sigma_y^2
        # i.e. twice the shrinkage coefficient, no factor of 2 in denominator.
        step_size = (1.0 - alpha_bar) / (2 * self.sigma_y ** 2)

        x_corrected = x_t.detach()
        for _ in range(self.N_gd):
            x_corrected = x_corrected.requires_grad_(True)
            difference = measurement - self.operator.forward(x_corrected, **kwargs)
            sq_norm = torch.linalg.norm(difference) ** 2
            sq_norm_grad = torch.autograd.grad(outputs=sq_norm, inputs=x_corrected)[0]
            with torch.no_grad():
                # grad_norm = sq_norm_grad.norm(2).clamp(min=1e-8)
                x_corrected = x_corrected - self.lr * step_size * sq_norm_grad # / grad_norm

        distance = torch.sqrt(sq_norm)

        return x_corrected, distance


# ---------------------------------------------------------------------------
# FRITES sampler
# ---------------------------------------------------------------------------

@register_sampler(name='frites')
class FRITESampler(SpacedDiffusion):
    """
    Iterative FRITES algorithm (lines 342-414 in main_gmm_nuts.py).

    Outer loop over sigma schedule.  At each level gamma = sigma:

      1. FRITES — gradient step (VP-SDE, DDPM-scheduled alpha_bar):
            x_corrected = x - ((1-alpha_bar)/(2*sigma_y^2)) * grad(||y - A(x)||^2)

      2. Forward diffuse — DDPM-scheduled noise injection:
            x_noisy = sqrt(alpha_bar)*x_before + (x_corrected - x_before) + sqrt(1-alpha_bar)*eps
            i.e.    = sqrt(alpha_bar)*x + (1-alpha_bar)*grad_Lt + sqrt(1-alpha_bar)*eps

      3. DDS — DDPM reverse chain for prior consistency (DDPMDenoiser):
            x = DDPMDenoiser.denoise(x_noisy, gamma, model)

    Config params (under diffusion_config['params'])
    -------------------------------------------------
    dt       : float, default 0.01.  Step size for the forward SDE, controls the strength of the measurement consistency gradient step.
    K        : int, default 100.   Number of iterations to run the forward S
    """

    def p_sample_loop(
        self,
        model,
        x_start,
        measurement,
        measurement_cond_fn,
        record,
        save_root,
        ref_img=None,
        fname=None,
        **kwargs,
    ):
        device = x_start.device

        # ------------------------------------------------------------------
        # warm start ------------------------------------------------------------------

        if self.kwargs.get('warm_start', False):
            std_config = {
                "sampler": 'ddpm',
                "steps": 1000,
                "noise_schedule": 'linear',
                "model_mean_type": 'epsilon',
                "model_var_type": 'learned_range',
                "dynamic_threshold": False,
                "clip_denoised": True,
                "rescale_timesteps": False,
                "timestep_respacing": 500
            }
            default_sampler = create_sampler(adjust_color=adjust_color, **std_config) 
            default_fn = partial(
                default_sampler.p_sample_loop, 
                model=model, 
                measurement_cond_fn=kwargs['default_cond_fn'])
            img = default_fn(
                x_start=x_start, 
                measurement=measurement, 
                record=False, 
                save_root=save_root)
        else:
            img = x_start

        
        plt.imsave(
            os.path.join(save_root, 'init', fname[:5] + 'init.png'),
            clear_color(img),
        )

        # dt variant: one small OU forward step of size dt, then reverse to t=0.
        # sigma_dt = sqrt(exp(2*dt) - 1) is the noise level after one dt step.
        # alpha_bar is looked up once from the DDPM schedule and held constant.
        dt       = kwargs.get('dt', 0.01)
        K        = kwargs.get('K', 100)
        n_ddim   = kwargs.get('n_ddim', 5)
        eta      = kwargs.get('eta', 0.0)

        sigma_dt = math.sqrt(math.exp(2.0 * dt) - 1.0)
        alpha_bar_target = 1.0 / (1.0 + sigma_dt ** 2)
        time_mask = self.alphas_cumprod > alpha_bar_target
        if time_mask.any():
            alpha_bar = float(self.alphas_cumprod[time_mask][-1])
        else:
            alpha_bar = alpha_bar_target

        denoiser = DDPMDenoiser(self)

        x_0 = img
        x_0_hat = img

        # Forward-diffuse x_start once so the first DDS step has a noisy input.
        # with torch.no_grad():
        #     eps = torch.randn_like(img)
        #     x_noisy = math.sqrt(alpha_bar) * img + math.sqrt(1.0 - alpha_bar) * eps

        if record:
            psnr_history = []
            lpips_history = []
            loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

        pbar = tqdm(range(K))
        for idx in pbar:
            sigma_torch = torch.tensor([sigma_dt] * img.shape[0], device=device)

            # ----------------------------------------------------------
            # Step 1: Gradient step — measurement consistency at x_0_hat.
            # ----------------------------------------------------------
            x_0_hat = x_0_hat.requires_grad_(True)
            x_corrected, distance = measurement_cond_fn(
                x_t=x_0_hat,
                measurement=measurement,
                noisy_measurement=measurement,
                gamma=sigma_dt,
                alpha_bar=alpha_bar,
            )
            x_corrected = x_corrected.detach()
            # x_0 = x_0.detach()

            # ----------------------------------------------------------
            # Step 2: Forward diffuse from x_0_hat.
            eps = torch.randn_like(x_corrected)
            x_noisy = math.sqrt(alpha_bar) * x_corrected + math.sqrt(1.0 - alpha_bar) * eps

            # eps = torch.randn_like(x_0)
            # x_noisy = math.sqrt(alpha_bar) * x_0 + (x_corrected - x_0_hat) + math.sqrt(1.0 - alpha_bar) * eps

            if record and idx % 10 == 0:
                plt.imsave(
                    os.path.join(save_root, f"progress/forward_{str(idx).zfill(4)}.png"),
                    clear_color(x_noisy),
                )

            pbar.set_postfix({'distance': distance.item()}, refresh=False)

            # ----------------------------------------------------------
            # Step 3: DDS — get clean estimate x_0 for this iteration.
            # ----------------------------------------------------------
            with torch.no_grad():
                x_0, x_0_hat = denoiser.denoise_ddim(
                    noise_lvl=sigma_torch, model=model, x_noisy=x_noisy,
                    n_ddim=n_ddim, eta=eta,
                )

                if record and ref_img is not None:
                    with torch.no_grad():
                        psnr_val = peak_signal_noise_ratio(
                            ref_img.detach().cpu().numpy(),
                            x_0_hat.detach().cpu().numpy()
                        )
                        lpips_val = loss_fn_vgg(x_0_hat, ref_img).item()
                    psnr_history.append(psnr_val)
                    lpips_history.append(lpips_val)
                    pbar.set_postfix({'distance': distance.item(), 'psnr': f'{psnr_val:.2f}', 'lpips': f'{lpips_val:.4f}'}, refresh=False)

                # x_0, x_0_hat = denoiser.denoise_one_step(
                #     noise_lvl=sigma_torch, model=model, x_noisy=x_noisy
                # )
                # x_0 = denoiser.denoise(noise_lvl=sigma_torch, model=model, x_noisy=x_noisy)


            # Verify consistency
            # denoiser.gen_schedule(sigma_torch)
            # t_idx = len(denoiser.ts) - 1
            # alpha_bar_denoiser = float(self.alphas_cumprod[t_idx])
            # print(f"forward alpha_bar: {alpha_bar:.6f}  |  denoiser alpha_bar: {alpha_bar_denoiser:.6f}")
            # exit()

            if self.adjust_color and idx == 2:
                x_0 = adjust_color(x_0, model, tol=sigma_torch)

            if record and idx % 10 == 0:
                plt.imsave(
                    os.path.join(save_root, f"progress/reverse_{str(idx).zfill(4)}.png"),
                    clear_color(x_0_hat),
                )
                # plt.imsave(
                #     os.path.join(save_root, f"progress/reverse_clean_{str(idx).zfill(4)}.png"),
                #     clear_color(x_0_hat),
                # )

            # prepared_init_img = prepare_im(os.path.join(out_path, 'progress', "init.png"), 256, device=device)
            # psnr_normal_init = peak_signal_noise_ratio(ref_img.detach().cpu().numpy(), prepared_init_img.detach().cpu().numpy())
            # normal_d_init = loss_fn_vgg(prepared_init_img, prepared_ref_img)


            if idx == K - 1:
                if record and ref_img is not None and len(psnr_history) > 0:
                    iters = list(range(len(psnr_history)))
                    np.save(os.path.join(save_root, 'progress', 'psnr_history.npy'), np.array(psnr_history))
                    np.save(os.path.join(save_root, 'progress', 'lpips_history.npy'), np.array(lpips_history))

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(iters, psnr_history)
                    ax1.set_xlabel('Iteration'); ax1.set_ylabel('PSNR (dB)'); ax1.set_title('PSNR vs Iteration')
                    ax2.plot(iters, lpips_history)
                    ax2.set_xlabel('Iteration'); ax2.set_ylabel('LPIPS'); ax2.set_title('LPIPS vs Iteration')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_root, 'recon', fname.replace('.png', '_metrics.png')), dpi=150)
                    plt.close()

                return x_0_hat

        return x_0_hat
            
        # x_0 = img          # hidden EM state (noisy after first iteration)
        # x_0_hat = img      # clean estimate for gradient (pred_xstart of x_0)

        # pbar = tqdm(range(K))
        # for idx in pbar:
        #     sigma_torch = torch.tensor([sigma_dt] * img.shape[0], device=device)

        #     # Step 1: Gradient at CLEAN estimate, not noisy EM state
        #     x_0_hat = x_0_hat.requires_grad_(True)
        #     x_corrected, distance = measurement_cond_fn(
        #         x_t=x_0_hat,
        #         measurement=measurement,
        #         noisy_measurement=measurement,
        #         gamma=sigma_dt,
        #         alpha_bar=alpha_bar,
        #     )
        #     x_0_hat = x_0_hat.detach()

        #     # Step 2: Forward diffuse clean corrected image
        #     eps = torch.randn_like(x_corrected)
        #     x_noisy = math.sqrt(alpha_bar) * x_corrected + math.sqrt(1.0 - alpha_bar) * eps

        #     pbar.set_postfix({'distance': distance.item()}, refresh=False)

        #     # Step 3: EM step — returns noisy hidden state AND clean estimate for next gradient step
        #     with torch.no_grad():
        #         x_0, x_0_hat = denoiser.denoise_one_step(
        #             noise_lvl=sigma_torch, model=model, x_noisy=x_noisy
        #         )

        #     if idx == K - 1:
        #         return x_0_hat  # return clean estimate, not noisy EM state


        # return x_0_hat
