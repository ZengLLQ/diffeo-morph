"""Vector field regularization and Hilbert space Sobolev smoothing."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


class VectorRegularizer:
    """Applies Gaussian and fluid-like Helmholtz smoothing to velocity fields."""

    @staticmethod
    def smooth_velocity_field(v: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """
        Apply spatial Gaussian filter to each spatial component of velocity field v.
        Ensures velocity field belongs to reproducing kernel Hilbert space (RKHS) V.
        """
        ndim = v.shape[0]
        v_smoothed = np.zeros_like(v)
        for d in range(ndim):
            v_smoothed[d] = gaussian_filter(v[d], sigma=sigma)
        return v_smoothed

    @staticmethod
    def diffusion_energy(v: np.ndarray) -> float:
        """
        Compute Dirichlet diffusion energy: 0.5 * sum_d ||grad(v_d)||^2.
        """
        ndim = v.shape[0]
        total_energy = 0.0
        for d in range(ndim):
            grads = np.gradient(v[d])
            for g in grads:
                total_energy += float(np.sum(g ** 2))
        return 0.5 * total_energy / v.size
