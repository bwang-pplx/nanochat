"""
Noise schedule and corruption functions for absorbing discrete diffusion (SEDD).

The absorbing process replaces tokens with a MASK token independently with
probability 1 - exp(-sigma_bar(t)), where sigma_bar(t) is a monotonically
increasing noise schedule.

Reference: Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios
of the Data Distribution", ICML 2024.
"""

import torch


def get_sigma_bar(t, schedule='loglinear', sigma_min=1e-4, sigma_max=20.0):
    """
    Noise schedule: cumulative noise level at time t in [0, 1].

    Args:
        t: Tensor of shape (B,) with values in [0, 1].
        schedule: 'geometric' or 'loglinear'.
        sigma_min: Minimum noise level (geometric schedule).
        sigma_max: Maximum noise level (geometric schedule).

    Returns:
        sigma_bar: Tensor of shape (B,) with noise levels.
    """
    if schedule == 'geometric':
        return sigma_min ** (1 - t) * sigma_max ** t
    elif schedule == 'loglinear':
        eps = 1e-3
        return -torch.log1p(-(1 - eps) * t)
    else:
        raise ValueError(f"Unknown noise schedule: {schedule}")


def get_dsigma_dt(t, schedule='loglinear', sigma_min=1e-4, sigma_max=20.0):
    """
    Derivative of sigma_bar with respect to t, used for loss weighting.

    The denoising score entropy is a time integral approximated by Monte Carlo:
        L = E_t[ dsigma/dt * L(t) ]
    When t ~ Uniform, each sample must be weighted by dsigma/dt.

    Returns:
        dsigma_dt: Tensor of shape (B,).
    """
    if schedule == 'geometric':
        import math
        sigma_bar = sigma_min ** (1 - t) * sigma_max ** t
        return sigma_bar * math.log(sigma_max / sigma_min)
    elif schedule == 'loglinear':
        eps = 1e-3
        return (1 - eps) / (1 - (1 - eps) * t)
    else:
        raise ValueError(f"Unknown noise schedule: {schedule}")


def sample_time(batch_size, device, eps=1e-3):
    """Sample t ~ Uniform(eps, 1) per batch element."""
    return eps + (1 - eps) * torch.rand(batch_size, device=device)


def corrupt_absorbing(x0, t, mask_token_id, schedule='loglinear', sigma_min=1e-4, sigma_max=20.0):
    """
    Forward (corruption) process: replace each token with MASK independently.

    Each token is replaced with probability 1 - exp(-sigma_bar(t)).

    Args:
        x0: Clean token ids, shape (B, T).
        t: Time values, shape (B,).
        mask_token_id: Integer id for the MASK token.
        schedule, sigma_min, sigma_max: Noise schedule parameters.

    Returns:
        x_t: Corrupted token ids, shape (B, T).
        is_masked: Boolean mask, shape (B, T). True where tokens are masked.
    """
    sigma_bar = get_sigma_bar(t, schedule, sigma_min, sigma_max)  # (B,)
    # Probability of keeping each token = exp(-sigma_bar)
    keep_prob = torch.exp(-sigma_bar).unsqueeze(1)  # (B, 1)
    # Sample which tokens to keep
    rand = torch.rand_like(x0, dtype=keep_prob.dtype)
    is_masked = rand >= keep_prob  # True where masked
    x_t = torch.where(is_masked, mask_token_id, x0)
    return x_t, is_masked


def stable_expm1(sigma):
    """Compute exp(sigma) - 1 with numerical stability for small sigma."""
    return torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)


def absorbing_ratio(sigma_bar):
    """
    Denoising ratio used in the score entropy loss:
        r = exp(-sigma_bar) / (1 - exp(-sigma_bar)) = 1 / (exp(sigma_bar) - 1)

    Uses torch.expm1 for numerical stability when sigma_bar is small.

    Args:
        sigma_bar: Tensor of noise levels.

    Returns:
        ratio: Tensor of same shape.
    """
    return 1.0 / stable_expm1(sigma_bar)
