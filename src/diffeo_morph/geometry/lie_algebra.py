"""Stationary Velocity Field (SVF) Lie Group Scaling and Squaring."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from diffeo_morph.geometry.grid import create_identity_grid, compose_transformations


def scaling_and_squaring(
    velocity_field: np.ndarray,
    n_steps: int = 7,
    compute_inverse: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Compute diffeomorphism phi = exp(v) via Lie group scaling and squaring.
    
    Args:
        velocity_field: Stationary velocity field of shape (ndim, *shape).
        n_steps: Number of squaring steps (2^n_steps scale factor, default 7 -> 128).
        compute_inverse: If True, also returns the inverse diffeomorphism phi_inv = exp(-v).
    Returns:
        phi_forward: exp(v) of shape (ndim, *shape).
        phi_inverse: exp(-v) of shape (ndim, *shape) or None.
    """
    ndim = velocity_field.shape[0]
    shape = velocity_field.shape[1:]
    id_grid = create_identity_grid(shape)

    scale_factor = 2.0 ** n_steps
    v_scaled = velocity_field / scale_factor

    # Small displacement approximation: exp(v / 2^N) ~ Id + v / 2^N
    phi = id_grid + v_scaled

    # Repeated squaring: phi = phi o phi
    for _ in range(n_steps):
        phi = compose_transformations(phi, phi, order=1)

    phi_inv = None
    if compute_inverse:
        v_inv_scaled = -velocity_field / scale_factor
        phi_inv = id_grid + v_inv_scaled
        for _ in range(n_steps):
            phi_inv = compose_transformations(phi_inv, phi_inv, order=1)

    return phi, phi_inv
