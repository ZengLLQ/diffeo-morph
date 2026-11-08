"""Unit tests for similarity metrics and variational loss gradients."""

import numpy as np
import pytest
from diffeo_morph.registration import (
    mean_squared_error,
    local_normalized_cross_correlation,
)


def test_mean_squared_error():
    a = np.ones((10, 10))
    loss_zero, grad_zero = mean_squared_error(a, a)
    assert loss_zero == pytest.approx(0.0)
    np.testing.assert_allclose(grad_zero, 0.0)

    b = a + 2.0
    loss, grad = mean_squared_error(a, b)
    assert loss == pytest.approx(4.0)
    assert np.all(grad < 0.0)


def test_local_normalized_cross_correlation():
    img = np.random.RandomState(42).rand(16, 16)
    loss, grad = local_normalized_cross_correlation(img, img)
    # Identical images produce minimum loss (1 - 1 = 0)
    assert loss == pytest.approx(0.0, abs=1e-3)
