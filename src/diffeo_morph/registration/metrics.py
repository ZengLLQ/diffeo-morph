"""Image similarity metrics and variational loss gradients for deformable registration."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def mean_squared_error(
    moving_warped: np.ndarray,
    fixed: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute Mean Squared Error (MSE) and its analytical spatial gradient with respect to moving_warped.
    
    Returns:
        loss: Scalar MSE value.
        grad: Array of same shape as moving_warped, d(MSE) / d(I_warped).
    """
    diff = moving_warped - fixed
    loss = float(np.mean(diff ** 2))
    grad = (2.0 / diff.size) * diff
    return loss, grad


def local_normalized_cross_correlation(
    moving_warped: np.ndarray,
    fixed: np.ndarray,
    radius: int = 2,
    eps: float = 1e-5,
) -> Tuple[float, np.ndarray]:
    """
    Compute Local Normalized Cross-Correlation (LNCC) for contrast-invariant MRI registration.
    
    Returns:
        loss: 1.0 - mean(LNCC) (minimized at 0 when perfectly correlated).
        grad: Analytical gradient d(loss) / d(I_warped).
    """
    # Box size
    size = 2 * radius + 1

    # Local means
    u_i = uniform_filter(moving_warped, size=size)
    u_j = uniform_filter(fixed, size=size)

    # Local cross terms
    u_i2 = uniform_filter(moving_warped ** 2, size=size)
    u_j2 = uniform_filter(fixed ** 2, size=size)
    u_ij = uniform_filter(moving_warped * fixed, size=size)

    # Local variances and covariance
    var_i = np.maximum(0.0, u_i2 - u_i ** 2)
    var_j = np.maximum(0.0, u_j2 - u_j ** 2)
    cov_ij = u_ij - u_i * u_j

    denom = np.sqrt(var_i * var_j) + eps
    cc = cov_ij / denom

    loss = float(1.0 - np.mean(cc))

    # Analytical variational gradient
    grad = (
        (cov_ij / (var_i + eps)) * (moving_warped - u_i) - (fixed - u_j)
    ) / (denom * moving_warped.size)

    return loss, grad
