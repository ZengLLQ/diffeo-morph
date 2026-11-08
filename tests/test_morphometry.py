"""Unit tests for Tensor-Based Morphometry (TBM) and Alzheimer Clinical Staging."""

import numpy as np
import pytest
from diffeo_morph.clinical import (
    RegionalMorphometryMetrics,
    TensorBasedMorphometry,
    AlzheimerEvaluator,
)


def test_tbm_roi_analysis():
    # Synthetic 16x16 Jacobian map
    jac = np.ones((16, 16))
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True  # 16 voxels

    # In ROI, set local contraction |J| = 0.85 (15% volume atrophy)
    jac[mask] = 0.85

    metrics = TensorBasedMorphometry.analyze_roi(jac, mask, "Hippocampus")
    assert metrics.baseline_voxels == 16
    assert metrics.deformed_volume_voxels == pytest.approx(16 * 0.85)
    assert metrics.volume_ratio == pytest.approx(0.85)
    assert metrics.percent_volume_change == pytest.approx(-15.0)
    assert metrics.mean_log_jacobian < 0.0


def test_alzheimer_evaluator_staging():
    # Hippocampal loss: -6.0% / year, Ventricle expansion: +8.0% / year -> Probable AD
    hippo = RegionalMorphometryMetrics(
        region_name="Hippocampus",
        baseline_voxels=100,
        deformed_volume_voxels=94.0,
        volume_ratio=0.94,
        percent_volume_change=-6.0,
        mean_jacobian=0.94,
        mean_log_jacobian=-0.062,
        min_jacobian=0.90,
        max_jacobian=0.98,
    )
    vent = RegionalMorphometryMetrics(
        region_name="Ventricles",
        baseline_voxels=200,
        deformed_volume_voxels=216.0,
        volume_ratio=1.08,
        percent_volume_change=+8.0,
        mean_jacobian=1.08,
        mean_log_jacobian=0.077,
        min_jacobian=1.02,
        max_jacobian=1.14,
    )

    report = AlzheimerEvaluator.evaluate_longitudinal_scan(hippo, vent, scan_interval_months=12.0)
    assert report.clinical_stage == "PROBABLE_ALZHEIMERS_DISEASE"
    assert report.confidence_score > 0.80
