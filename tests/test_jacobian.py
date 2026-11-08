"""Unit tests for Jacobian matrix, determinant, and topological folding detection."""

import numpy as np
import pytest
from diffeo_morph.geometry import (
    create_identity_grid,
    compute_jacobian_determinant,
    count_foldings,
    compute_log_jacobian,
)


def test_identity_grid_jacobian():
    id_grid_2d = create_identity_grid((16, 16))
    det_2d = compute_jacobian_determinant(id_grid_2d)
    
    # In interior voxels, Jacobian determinant of identity grid is 1.0
    np.testing.assert_allclose(det_2d[1:-1, 1:-1], 1.0, atol=1e-4)
    assert count_foldings(det_2d) == 0

    id_grid_3d = create_identity_grid((8, 8, 8))
    det_3d = compute_jacobian_determinant(id_grid_3d)
    np.testing.assert_allclose(det_3d[1:-1, 1:-1, 1:-1], 1.0, atol=1e-4)
    assert count_foldings(det_3d) == 0


def test_folding_detection():
    # Construct an inverted/folded deformation field
    folded_grid = create_identity_grid((12, 12))
    # Invert x coordinates in center to produce negative Jacobian determinant
    folded_grid[1, :, 4:8] = -folded_grid[1, :, 4:8]

    det = compute_jacobian_determinant(folded_grid)
    fold_count = count_foldings(det)
    assert fold_count > 0


def test_log_jacobian():
    det = np.array([[0.5, 1.0], [2.0, 4.0]])
    log_j = compute_log_jacobian(det)
    expected = np.array([[np.log(0.5), 0.0], [np.log(2.0), np.log(4.0)]])
    np.testing.assert_allclose(log_j, expected, atol=1e-4)
