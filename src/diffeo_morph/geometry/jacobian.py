"""Jacobian determinant and topological folding analysis for diffeomorphic maps."""

from __future__ import annotations
from typing import Tuple
import numpy as np


def compute_jacobian_matrix(phi: np.ndarray) -> np.ndarray:
    """
    Compute spatial gradient of deformation field D phi.
    
    Args:
        phi: Mapping field of shape (ndim, D1, D2, ...)
    Returns:
        Tensor of shape (ndim, ndim, D1, D2, ...) where J[i, j] = d(phi_i) / d(x_j).
    """
    ndim = phi.shape[0]
    shape = phi.shape[1:]
    jac = np.zeros((ndim, ndim, *shape), dtype=np.float64)

    for i in range(ndim):
        grads = np.gradient(phi[i])
        for j in range(ndim):
            jac[i, j] = grads[j]

    return jac


def compute_jacobian_determinant(phi: np.ndarray) -> np.ndarray:
    """
    Compute point-wise Jacobian determinant |J_phi(x)|.
    
    Values > 1 indicate local tissue expansion.
    Values < 1 indicate local tissue contraction / atrophy.
    Values <= 0 indicate topological singularity / grid folding (violating diffeomorphism).
    """
    ndim = phi.shape[0]
    shape = phi.shape[1:]
    jac = compute_jacobian_matrix(phi)

    if ndim == 2:
        j11 = jac[0, 0]
        j12 = jac[0, 1]
        j21 = jac[1, 0]
        j22 = jac[1, 1]
        det = j11 * j22 - j12 * j21
    elif ndim == 3:
        j11, j12, j13 = jac[0, 0], jac[0, 1], jac[0, 2]
        j21, j22, j23 = jac[1, 0], jac[1, 1], jac[1, 2]
        j31, j32, j33 = jac[2, 0], jac[2, 1], jac[2, 2]

        det = (
            j11 * (j22 * j33 - j23 * j32)
            - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31)
        )
    else:
        n_voxels = int(np.prod(shape))
        flat_jac = np.moveaxis(jac, [0, 1], [-2, -1]).reshape(n_voxels, ndim, ndim)
        det = np.linalg.det(flat_jac).reshape(shape)

    return det


def count_foldings(jacobian_det: np.ndarray) -> int:
    """Return count of voxels with negative or zero Jacobian determinant."""
    return int(np.sum(jacobian_det <= 0.0))


def compute_log_jacobian(jacobian_det: np.ndarray, clip_min: float = 1e-4) -> np.ndarray:
    """
    Compute log-Jacobian map log(|J_phi(x)|).
    Used extensively in Tensor-Based Morphometry (TBM).
    """
    clipped = np.maximum(clip_min, jacobian_det)
    return np.log(clipped)
