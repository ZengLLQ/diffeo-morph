"""Unit tests for coordinate grids, image warping, and transformation composition."""

import numpy as np
import pytest
from diffeo_morph.geometry import (
    create_identity_grid,
    warp_image,
    compose_transformations,
)


def test_create_identity_grid_2d_and_3d():
    grid_2d = create_identity_grid((16, 16))
    assert grid_2d.shape == (2, 16, 16)
    assert grid_2d[0, 0, 0] == 0.0
    assert grid_2d[0, 15, 0] == 15.0
    assert grid_2d[1, 0, 15] == 15.0

    grid_3d = create_identity_grid((8, 8, 8))
    assert grid_3d.shape == (3, 8, 8, 8)


def test_warp_image_identity():
    img = np.random.RandomState(42).rand(16, 16)
    id_grid = create_identity_grid((16, 16))
    warped = warp_image(img, id_grid, is_displacement=False, order=1)
    np.testing.assert_allclose(warped, img, atol=1e-5)


def test_warp_image_displacement_shift():
    img = np.zeros((16, 16))
    img[5:10, 5:10] = 1.0

    # Shift displacement by +2 in y and +3 in x
    disp = np.zeros((2, 16, 16))
    disp[0] = 2.0
    disp[1] = 3.0

    warped = warp_image(img, disp, is_displacement=True, order=1)
    # The active square in warped should now be sampled from (y+2, x+3)
    assert warped[3, 2] == pytest.approx(1.0, abs=1e-2)


def test_compose_transformations_identity():
    id_grid = create_identity_grid((12, 12))
    composed = compose_transformations(id_grid, id_grid)
    np.testing.assert_allclose(composed, id_grid, atol=1e-4)
