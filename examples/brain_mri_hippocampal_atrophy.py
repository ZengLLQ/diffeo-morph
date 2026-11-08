#!/usr/bin/env python3
"""Clinical Example: Longitudinal 3D Brain MRI Hippocampal Atrophy Quantification."""

import numpy as np
from diffeo_morph.registration import SVFRegistrationOptimizer
from diffeo_morph.clinical import TensorBasedMorphometry, AlzheimerEvaluator
from diffeo_morph.visualizer import render_clinical_report


def run_hippocampal_atrophy_example():
    print("Executing Clinical Alzheimer's Longitudinal Morphometry Analysis...")

    # Create synthetic 3D baseline and 1-year follow-up brain volumes
    dim = 24
    shape = (dim, dim, dim)
    
    baseline = np.zeros(shape, dtype=np.float64)
    followup = np.zeros(shape, dtype=np.float64)
    hippo_mask = np.zeros(shape, dtype=bool)
    vent_mask = np.zeros(shape, dtype=bool)

    z, y, x = np.ogrid[:dim, :dim, :dim]
    center = dim / 2.0

    # Brain parenchyma sphere
    parenchyma = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= (dim * 0.42)**2
    baseline[parenchyma] = 0.70
    followup[parenchyma] = 0.70

    # Ventricles: expanding in follow-up
    r_v_base = dim * 0.10
    r_v_follow = dim * 0.13
    vent_base = ((x - center)**2 + (y - center)**2 + ((z - center)*1.2)**2) <= (r_v_base**2)
    vent_follow = ((x - center)**2 + (y - center)**2 + ((z - center)*1.2)**2) <= (r_v_follow**2)
    
    baseline[vent_base] = 0.15
    followup[vent_follow] = 0.15
    vent_mask[vent_base] = True

    # Hippocampus: contracting in follow-up (Alzheimer's atrophy)
    r_h_base = dim * 0.08
    r_h_follow = dim * 0.06
    h_base = ((x - (center - dim*0.2))**2 + (y - center)**2 + (z - center)**2) <= (r_h_base**2)
    h_follow = ((x - (center - dim*0.2))**2 + (y - center)**2 + (z - center)**2) <= (r_h_follow**2)

    baseline[h_base] = 0.45
    followup[h_follow] = 0.45
    hippo_mask[h_base] = True

    # Register follow-up to baseline
    optimizer = SVFRegistrationOptimizer(learning_rate=0.4, fluid_sigma=1.2)
    res = optimizer.register(followup, baseline, num_iterations=25)

    hippo_stats = TensorBasedMorphometry.analyze_roi(res.jacobian_determinant, hippo_mask, "Hippocampus")
    vent_stats = TensorBasedMorphometry.analyze_roi(res.jacobian_determinant, vent_mask, "Lateral Ventricles")

    report = AlzheimerEvaluator.evaluate_longitudinal_scan(
        hippo_stats,
        vent_stats,
        scan_interval_months=12.0,
    )

    print(render_clinical_report(report, hippo_stats, vent_stats))

    assert res.num_foldings == 0
    assert hippo_stats.percent_volume_change != 0.0
    print("SUCCESS: 3D Brain MRI morphometric evaluation completed without errors.")


if __name__ == "__main__":
    run_hippocampal_atrophy_example()
