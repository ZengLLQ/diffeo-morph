"""Diffeomorphic geometry, Lie algebra, and spatial transformation tools."""

from diffeo_morph.geometry.grid import (
    create_identity_grid,
    warp_image,
    compose_transformations,
)
from diffeo_morph.geometry.lie_algebra import scaling_and_squaring
from diffeo_morph.geometry.jacobian import (
    compute_jacobian_matrix,
    compute_jacobian_determinant,
    count_foldings,
    compute_log_jacobian,
)

__all__ = [
    "create_identity_grid",
    "warp_image",
    "compose_transformations",
    "scaling_and_squaring",
    "compute_jacobian_matrix",
    "compute_jacobian_determinant",
    "count_foldings",
    "compute_log_jacobian",
]
