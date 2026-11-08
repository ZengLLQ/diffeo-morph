"""Unit tests for non-rigid diffeomorphic registration optimizer."""

import numpy as np
import pytest
from diffeo_morph.registration import SVFRegistrationOptimizer


def test_2d_registration_convergence():
    dim = 20
    moving = np.zeros((dim, dim))
    fixed = np.zeros((dim, dim))

    # Moving has a square at [6:12, 6:12]
    moving[6:12, 6:12] = 1.0
    # Fixed has a square shifted slightly to [7:13, 7:13]
    fixed[7:13, 7:13] = 1.0

    optimizer = SVFRegistrationOptimizer(learning_rate=0.5, fluid_sigma=1.2)
    res = optimizer.register(moving, fixed, num_iterations=20)

    # Loss must decrease
    assert res.loss_history[-1] < res.loss_history[0]
    # Diffeomorphic condition: no topological folding
    assert res.num_foldings == 0
    assert res.is_diffeomorphic


def test_3d_registration():
    dim = 12
    moving = np.zeros((dim, dim, dim))
    fixed = np.zeros((dim, dim, dim))

    moving[4:8, 4:8, 4:8] = 1.0
    fixed[5:9, 5:9, 5:9] = 1.0

    optimizer = SVFRegistrationOptimizer(learning_rate=0.4, fluid_sigma=1.0)
    res = optimizer.register(moving, fixed, num_iterations=10)

    assert res.loss_history[-1] < res.loss_history[0]
    assert res.num_foldings == 0
