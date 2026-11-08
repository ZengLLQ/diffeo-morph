"""Unit tests for Lie group scaling and squaring of stationary velocity fields."""

import numpy as np
import pytest
from diffeo_morph.geometry import (
    create_identity_grid,
    compose_transformations,
    scaling_and_squaring,
)


def test_scaling_and_squaring_zero_velocity():
    shape = (10, 10)
    v_zero = np.zeros((2, *shape))
    phi, phi_inv = scaling_and_squaring(v_zero, n_steps=5, compute_inverse=True)

    id_grid = create_identity_grid(shape)
    np.testing.assert_allclose(phi, id_grid, atol=1e-5)
    np.testing.assert_allclose(phi_inv, id_grid, atol=1e-5)


def test_scaling_and_squaring_inverse_consistency():
    # Mild smooth velocity field
    shape = (16, 16)
    v = np.zeros((2, *shape))
    y, x = np.ogrid[:16, :16]
    v[0] = 0.5 * np.sin(x / 3.0)
    v[1] = 0.5 * np.cos(y / 3.0)

    phi_fwd, phi_inv = scaling_and_squaring(v, n_steps=6, compute_inverse=True)
    id_grid = create_identity_grid(shape)

    # phi o phi_inv ~ Id in the interior domain
    composed = compose_transformations(phi_fwd, phi_inv)
    # Check interior voxels [3:13, 3:13] away from boundaries
    np.testing.assert_allclose(composed[:, 3:13, 3:13], id_grid[:, 3:13, 3:13], atol=0.15)
