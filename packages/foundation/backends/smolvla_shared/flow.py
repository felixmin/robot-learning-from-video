from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor is expected to be of shape (batch_size,)")

    dtype = torch.float64 if device.type == "mps" else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * torch.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
    beta: float,
) -> Tensor:
    beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
    time_beta = beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
    return time_beta * 0.999 + 0.001


def make_noisy_target(target: Tensor, noise: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    time_expanded = time.reshape(time.shape[0], *([1] * (target.ndim - 1)))
    x_t = time_expanded * noise + (1.0 - time_expanded) * target
    u_t = noise - target
    return x_t, u_t


def reverse_euler_integration(
    *,
    initial: Tensor,
    num_steps: int,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")

    x_t = initial
    dt = -1.0 / float(num_steps)
    batch_size = int(initial.shape[0])

    for step in range(num_steps):
        t_val = 1.0 + step * dt
        time = torch.full((batch_size,), t_val, dtype=initial.dtype, device=initial.device)
        v_t = velocity_fn(x_t, time)
        x_t = x_t + dt * v_t

    return x_t
