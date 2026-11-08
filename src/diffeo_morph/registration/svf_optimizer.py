"""Diffeomorphic Image Registration using Stationary Velocity Fields (SVF)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import zoom
from diffeo_morph.geometry.grid import create_identity_grid, warp_image
from diffeo_morph.geometry.lie_algebra import scaling_and_squaring
from diffeo_morph.geometry.jacobian import compute_jacobian_determinant, count_foldings
from diffeo_morph.registration.metrics import mean_squared_error, local_normalized_cross_correlation
from diffeo_morph.registration.regularizers import VectorRegularizer


@dataclass
class RegistrationResult:
    """Outcome of non-rigid diffeomorphic registration."""
    warped_image: np.ndarray
    velocity_field: np.ndarray  # Stationary velocity field v
    phi_forward: np.ndarray  # exp(v)
    phi_inverse: np.ndarray  # exp(-v)
    jacobian_determinant: np.ndarray
    num_foldings: int
    loss_history: List[float]

    @property
    def is_diffeomorphic(self) -> bool:
        return self.num_foldings == 0


class SVFRegistrationOptimizer:
    """Non-linear diffeomorphic registration via SVF gradient flow."""

    def __init__(
        self,
        learning_rate: float = 0.5,
        fluid_sigma: float = 1.5,
        elastic_weight: float = 0.01,
        metric: str = "mse",  # "mse" or "lncc"
    ):
        self.learning_rate = learning_rate
        self.fluid_sigma = fluid_sigma
        self.elastic_weight = elastic_weight
        self.metric = metric

    def register(
        self,
        moving: np.ndarray,
        fixed: np.ndarray,
        num_iterations: int = 50,
        n_squaring_steps: int = 6,
        initial_velocity: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        """
        Execute iterative diffeomorphic registration moving -> fixed.
        """
        shape = moving.shape
        ndim = moving.ndim

        if moving.shape != fixed.shape:
            raise ValueError(f"Moving {moving.shape} and fixed {fixed.shape} shapes must match")

        if initial_velocity is not None:
            v = np.copy(initial_velocity)
        else:
            v = np.zeros((ndim, *shape), dtype=np.float64)

        loss_history: List[float] = []

        for it in range(num_iterations):
            # Compute exp(v) and exp(-v)
            phi_fwd, phi_inv = scaling_and_squaring(v, n_steps=n_squaring_steps, compute_inverse=True)

            # Warp moving image using inverse diffeomorphism: I_warped = I_moving o exp(-v)
            warped = warp_image(moving, phi_inv, is_displacement=False, order=1)

            # Compute similarity metric loss and gradient w.r.t warped image
            if self.metric == "lncc":
                sim_loss, d_sim = local_normalized_cross_correlation(warped, fixed)
            else:
                sim_loss, d_sim = mean_squared_error(warped, fixed)

            reg_loss = self.elastic_weight * VectorRegularizer.diffusion_energy(v)
            total_loss = sim_loss + reg_loss
            loss_history.append(total_loss)

            # Spatial gradient of warped image: grad(I_warped)
            spatial_grads = np.gradient(warped)

            # Variational driving force to pull moving image towards fixed target
            diff = warped - fixed
            force = np.zeros_like(v)
            for d in range(ndim):
                force[d] = diff * spatial_grads[d]

            # Sobolev RKHS smoothing of update step (fluid regularization)
            smoothed_force = VectorRegularizer.smooth_velocity_field(force, sigma=self.fluid_sigma)

            # Update velocity field with step size
            v += self.learning_rate * smoothed_force

            # Penalize velocity magnitude (elastic shrinkage)
            if self.elastic_weight > 0.0:
                v *= (1.0 - self.elastic_weight * 0.05)

        # Final evaluation
        phi_fwd, phi_inv = scaling_and_squaring(v, n_steps=n_squaring_steps, compute_inverse=True)
        final_warped = warp_image(moving, phi_inv, is_displacement=False, order=1)
        jac_det = compute_jacobian_determinant(phi_fwd)
        foldings = count_foldings(jac_det)

        return RegistrationResult(
            warped_image=final_warped,
            velocity_field=v,
            phi_forward=phi_fwd,
            phi_inverse=phi_inv,
            jacobian_determinant=jac_det,
            num_foldings=foldings,
            loss_history=loss_history,
        )
