"""Command-Line Interface (CLI) for DiffeoMorph."""

from __future__ import annotations
import argparse
import sys
import numpy as np
from diffeo_morph.geometry import (
    create_identity_grid,
    warp_image,
    scaling_and_squaring,
    compute_jacobian_determinant,
    count_foldings,
)
from diffeo_morph.registration import SVFRegistrationOptimizer
from diffeo_morph.clinical import (
    TensorBasedMorphometry,
    AlzheimerEvaluator,
)
from diffeo_morph.visualizer import render_ascii_slice, render_clinical_report


def generate_synthetic_brain_slice(dim: int = 32, atrophy_factor: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic brain slice with hippocampus and lateral ventricle labels.
    """
    img = np.zeros((dim, dim), dtype=np.float64)
    hippo_mask = np.zeros((dim, dim), dtype=bool)
    ventricle_mask = np.zeros((dim, dim), dtype=bool)

    # Parenchyma (brain tissue circle)
    y, x = np.ogrid[:dim, :dim]
    center = dim / 2.0
    r_brain = dim * 0.42
    brain_circle = ((x - center) ** 2 + (y - center) ** 2) <= (r_brain ** 2)
    img[brain_circle] = 0.65

    # Ventricles (center oval)
    r_vent = dim * (0.08 + 0.03 * atrophy_factor)
    vent_circle = ((x - center) ** 2 + ((y - center) * 1.5) ** 2) <= (r_vent ** 2)
    img[vent_circle] = 0.1
    ventricle_mask[vent_circle] = True

    # Bilateral Hippocampus (two smaller lateral structures)
    r_hippo = max(1.5, dim * (0.06 - 0.02 * atrophy_factor))
    left_hippo = ((x - (center - dim * 0.2)) ** 2 + (y - (center + dim * 0.1)) ** 2) <= (r_hippo ** 2)
    right_hippo = ((x - (center + dim * 0.2)) ** 2 + (y - (center + dim * 0.1)) ** 2) <= (r_hippo ** 2)
    
    hippo_mask[left_hippo | right_hippo] = True
    img[left_hippo | right_hippo] = 0.45

    return img, hippo_mask, ventricle_mask


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="diffeo-morph",
        description="DiffeoMorph: LDDMM & Lie Group SVF 3D Brain MRI Morphometry Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Register
    reg_p = subparsers.add_parser("register", help="Execute non-rigid diffeomorphic registration")
    reg_p.add_argument("--size", type=int, default=24, help="Grid dimension (default: 24)")
    reg_p.add_argument("--iterations", type=int, default=30, help="Number of optimizer steps (default: 30)")
    reg_p.add_argument("--lr", type=float, default=0.4, help="Learning rate (default: 0.4)")
    reg_p.add_argument("--sigma", type=float, default=1.2, help="Fluid regularization sigma (default: 1.2)")

    # 2. Alzheimer Staging
    alz_p = subparsers.add_parser("alzheimer", help="Evaluate longitudinal Alzheimer's disease atrophy staging")
    alz_p.add_argument("--atrophy-severity", type=float, default=1.0, help="Atrophy progression severity 0.0 to 2.0 (default: 1.0)")
    alz_p.add_argument("--interval-months", type=float, default=12.0, help="Scan interval in months (default: 12.0)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "register":
        print("=" * 65)
        print("     DiffeoMorph: Stationary Velocity Field (SVF) Registration     ")
        print("=" * 65)
        moving, _, _ = generate_synthetic_brain_slice(dim=args.size, atrophy_factor=0.0)
        fixed, _, _ = generate_synthetic_brain_slice(dim=args.size, atrophy_factor=1.0)

        opt = SVFRegistrationOptimizer(learning_rate=args.lr, fluid_sigma=args.sigma)
        res = opt.register(moving, fixed, num_iterations=args.iterations)

        print(f"  Grid Size         : {args.size}x{args.size}")
        print(f"  Initial Loss      : {res.loss_history[0]:.6f}")
        print(f"  Final Loss        : {res.loss_history[-1]:.6f} (Reduced by {((res.loss_history[0] - res.loss_history[-1]) / res.loss_history[0]) * 100:.1f}%)")
        print(f"  Diffeomorphic     : {res.is_diffeomorphic} (Foldings: {res.num_foldings})")
        print(f"  Jacobian Det Range: [{np.min(res.jacobian_determinant):.3f}, {np.max(res.jacobian_determinant):.3f}]")
        print("-" * 65)
        print("Warped Deformed Brain Slice:")
        print(render_ascii_slice(res.warped_image))
        print("=" * 65)
        return 0

    elif args.command == "alzheimer":
        baseline_img, hippo_mask, vent_mask = generate_synthetic_brain_slice(dim=28, atrophy_factor=0.0)
        followup_img, _, _ = generate_synthetic_brain_slice(dim=28, atrophy_factor=args.atrophy_severity)

        opt = SVFRegistrationOptimizer(learning_rate=0.4, fluid_sigma=1.2)
        res = opt.register(followup_img, baseline_img, num_iterations=30)

        hippo_metrics = TensorBasedMorphometry.analyze_roi(res.jacobian_determinant, hippo_mask, "Hippocampus")
        vent_metrics = TensorBasedMorphometry.analyze_roi(res.jacobian_determinant, vent_mask, "Ventricles")

        report = AlzheimerEvaluator.evaluate_longitudinal_scan(
            hippo_metrics,
            vent_metrics,
            scan_interval_months=args.interval_months,
        )
        print(render_clinical_report(report, hippo_metrics, vent_metrics))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
