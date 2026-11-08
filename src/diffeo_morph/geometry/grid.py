"""Spatial coordinate grids, deformation composition, and image warping."""

from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
from scipy.ndimage import map_coordinates


def create_identity_grid(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generate identity coordinate grid.
    Returns:
        Array of shape (ndim, *shape) containing continuous voxel indices.
    """
    ndim = len(shape)
    coords = [np.arange(s, dtype=np.float64) for s in shape]
    mesh = np.meshgrid(*coords, indexing="ij")
    return np.stack(mesh, axis=0)


def warp_image(
    image: np.ndarray,
    displacement_or_phi: np.ndarray,
    is_displacement: bool = False,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Warp an N-D scalar image using deformation field or displacement field.
    
    Args:
        image: Array of shape (D1, D2, ...) or (H, W).
        displacement_or_phi: Array of shape (ndim, D1, D2, ...).
        is_displacement: If True, field represents u(x) where phi(x) = x + u(x).
                         If False, field represents absolute sampling coordinates phi(x).
        order: Spline interpolation order (0: nearest, 1: linear/bilinear/trilinear).
        mode: Boundary condition handling ("nearest", "constant", "reflect").
    """
    shape = image.shape
    ndim = image.ndim

    if displacement_or_phi.shape[0] != ndim:
        raise ValueError(f"Deformation channel dimension ({displacement_or_phi.shape[0]}) must match image ndim ({ndim})")

    if is_displacement:
        id_grid = create_identity_grid(shape)
        sample_coords = id_grid + displacement_or_phi
    else:
        sample_coords = displacement_or_phi

    warped = map_coordinates(
        image,
        sample_coords,
        order=order,
        mode=mode,
        cval=0.0,
        prefilter=False,
    )
    return warped


def compose_transformations(phi1: np.ndarray, phi2: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Compose two diffeomorphisms: phi_composed = phi1 o phi2.
    Resamples phi1 at coordinate locations specified by phi2.
    
    Args:
        phi1: First mapping of shape (ndim, *shape).
        phi2: Second mapping of shape (ndim, *shape).
    Returns:
        Composed mapping of shape (ndim, *shape).
    """
    ndim = phi1.shape[0]
    shape = phi1.shape[1:]
    composed = np.zeros_like(phi1)

    for d in range(ndim):
        composed[d] = map_coordinates(
            phi1[d],
            phi2,
            order=order,
            mode="nearest",
            prefilter=False,
        )
    return composed
