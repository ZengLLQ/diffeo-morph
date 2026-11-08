#!/usr/bin/env python3
"""Clinical Example: Atlas-Based Deformable Registration for Ventriculomegaly Assessment."""

import numpy as np
from diffeo_morph.registration import SVFRegistrationOptimizer
from diffeo_morph.clinical import TensorBasedMorphometry
from diffeo_morph.visualizer import render_ascii_slice


def run_ventriculomegaly_atlas_example():
    print("Executing Atlas-to-Patient Ventriculomegaly Registration...")

    dim = 28
    # Template atlas: Standard ventricle size
    atlas = np.zeros((dim, dim), dtype=np.float64)
    # Patient with Normal Pressure Hydrocephalus (NPH) marked ventricular enlargement
    patient = np.zeros((dim, dim), dtype=np.float64)
    vent_mask = np.zeros((dim, dim), dtype=bool)

    y, x = np.ogrid[:dim, :dim]
    center = dim / 2.0

    # Parenchyma
    atlas[((x - center)**2 + (y - center)**2) <= (dim * 0.42)**2] = 0.60
    patient[((x - center)**2 + (y - center)**2) <= (dim * 0.42)**2] = 0.60

    # Atlas ventricle
    vent_atlas = ((x - center)**2 + ((y - center) * 1.4)**2) <= (dim * 0.09)**2
    atlas[vent_atlas] = 0.15
    vent_mask[vent_atlas] = True

    # Patient enlarged ventricle
    vent_patient = ((x - center)**2 + ((y - center) * 1.4)**2) <= (dim * 0.18)**2
    patient[vent_patient] = 0.15

    # Register atlas to patient
    opt = SVFRegistrationOptimizer(learning_rate=0.5, fluid_sigma=1.4, metric="lncc")
    res = opt.register(atlas, patient, num_iterations=25)

    stats = TensorBasedMorphometry.analyze_roi(res.jacobian_determinant, vent_mask, "Ventricular System")

    print("=" * 65)
    print("      Atlas-Based Ventricular Expansion Quantification          ")
    print("=" * 65)
    print(f"  Ventricle Baseline Voxels : {stats.baseline_voxels}")
    print(f"  Deformed Volume Voxels    : {stats.deformed_volume_voxels:.1f}")
    print(f"  Expansion Ratio           : {stats.volume_ratio:.2f}x ({stats.percent_volume_change:+.1f}%)")
    print(f"  Mean log-Jacobian         : {stats.mean_log_jacobian:+.3f}")
    print(f"  Topology Folding Count    : {res.num_foldings} (Preserved Diffeomorphism)")
    print("-" * 65)
    print("Patient Target MRI Slice:")
    print(render_ascii_slice(patient, height=10, width=28))
    print("Warped Atlas Aligned to Patient:")
    print(render_ascii_slice(res.warped_image, height=10, width=28))
    print("=" * 65)

    assert res.num_foldings == 0
    assert stats.volume_ratio > 1.0  # Demonstrating ventricular expansion
    print("SUCCESS: Ventriculomegaly atlas registration completed successfully.")


if __name__ == "__main__":
    run_ventriculomegaly_atlas_example()
