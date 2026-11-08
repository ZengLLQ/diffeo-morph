"""Tensor-Based Morphometry (TBM) and regional volumetric deformation analysis."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from diffeo_morph.geometry.jacobian import compute_log_jacobian


@dataclass
class RegionalMorphometryMetrics:
    """Quantitative morphometric statistics for an anatomical region of interest (ROI)."""
    region_name: str
    baseline_voxels: int
    deformed_volume_voxels: float
    volume_ratio: float  # V_deformed / V_baseline
    percent_volume_change: float  # (volume_ratio - 1.0) * 100
    mean_jacobian: float
    mean_log_jacobian: float
    min_jacobian: float
    max_jacobian: float


class TensorBasedMorphometry:
    """Extracts voxel-wise and regional volume changes from deformation Jacobian fields."""

    @staticmethod
    def analyze_roi(
        jacobian_det: np.ndarray,
        roi_mask: np.ndarray,
        region_name: str = "ROI",
    ) -> RegionalMorphometryMetrics:
        """
        Evaluate morphometric expansion or atrophy within an anatomical binary mask.
        """
        if jacobian_det.shape != roi_mask.shape:
            raise ValueError(f"Shape mismatch: jacobian {jacobian_det.shape} vs mask {roi_mask.shape}")

        mask = roi_mask.astype(bool)
        n_vox = int(np.sum(mask))
        if n_vox == 0:
            raise ValueError(f"ROI mask for {region_name} contains 0 voxels")

        roi_jac = jacobian_det[mask]
        deformed_vol = float(np.sum(roi_jac))
        vol_ratio = deformed_vol / n_vox
        pct_change = (vol_ratio - 1.0) * 100.0

        log_jac = compute_log_jacobian(roi_jac)
        
        return RegionalMorphometryMetrics(
            region_name=region_name,
            baseline_voxels=n_vox,
            deformed_volume_voxels=deformed_vol,
            volume_ratio=vol_ratio,
            percent_volume_change=pct_change,
            mean_jacobian=float(np.mean(roi_jac)),
            mean_log_jacobian=float(np.mean(log_jac)),
            min_jacobian=float(np.min(roi_jac)),
            max_jacobian=float(np.max(roi_jac)),
        )

    @staticmethod
    def analyze_atlas_segmentation(
        jacobian_det: np.ndarray,
        segmentation_map: np.ndarray,
        label_names: Dict[int, str],
    ) -> Dict[str, RegionalMorphometryMetrics]:
        """Analyze all regions in a multi-label anatomical segmentation."""
        results = {}
        for label_id, name in label_names.items():
            mask = (segmentation_map == label_id)
            if np.sum(mask) > 0:
                results[name] = TensorBasedMorphometry.analyze_roi(jacobian_det, mask, region_name=name)
        return results
