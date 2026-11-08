"""ASCII 2D neuroimaging slice visualizer and morphometric report renderer."""

from __future__ import annotations
import numpy as np
from diffeo_morph.clinical.alzheimer_eval import AlzheimerClinicalReport
from diffeo_morph.clinical.morphometry import RegionalMorphometryMetrics


def render_ascii_slice(image_slice: np.ndarray, height: int = 14, width: int = 36) -> str:
    """
    Render 2D axial/coronal slice as ASCII intensity density map.
    """
    from scipy.ndimage import zoom
    
    # Resize slice to target height x width
    zoom_y = height / image_slice.shape[0]
    zoom_x = width / image_slice.shape[1]
    resampled = zoom(image_slice, (zoom_y, zoom_x), order=1)

    min_v = float(np.min(resampled))
    max_v = float(np.max(resampled))
    norm = (resampled - min_v) / (max_v - min_v + 1e-9)

    # 10-level ASCII density ramp
    ramp = " .:-=+*#%@"
    n_levels = len(ramp)

    lines = ["+" + "-" * width + "+"]
    for r in range(height):
        row_chars = []
        for c in range(width):
            val = norm[r, c]
            idx = int(round(val * (n_levels - 1)))
            idx = max(0, min(n_levels - 1, idx))
            row_chars.append(ramp[idx])
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)


def render_clinical_report(
    report: AlzheimerClinicalReport,
    hippo: RegionalMorphometryMetrics,
    ventricles: RegionalMorphometryMetrics,
) -> str:
    """Render clinical morphometric staging evaluation."""
    border = "=" * 70
    sub_border = "-" * 70

    out = [
        border,
        "    DiffeoMorph: Longitudinal Brain MRI Morphometry & TBM Report     ",
        border,
        "  REGIONAL MORPHOMETRIC ATROPHY ANALYSIS:",
        f"    Hippocampus Volume Change      : {hippo.percent_volume_change:+6.2f}% (Annualized: {-report.annualized_hippocampal_atrophy_rate:+6.2f}%/yr)",
        f"    Mean Hippocampal log|J|        : {hippo.mean_log_jacobian:+6.3f} (Local tissue contraction)",
        f"    Ventricular Expansion          : {ventricles.percent_volume_change:+6.2f}% (Ex-vacuo hydrocephalus)",
        f"    Mean Ventricular log|J|        : {ventricles.mean_log_jacobian:+6.3f}",
        sub_border,
        "  DIAGNOSTIC STAGING EVALUATION:",
        f"    Staging Classification         : {report.clinical_stage}",
        f"    Diagnostic Confidence          : {report.confidence_score * 100:.1f}%",
        f"    Pathology Summary              : {report.diagnostic_summary}",
        sub_border,
        "  CLINICAL MANAGEMENT RECOMMENDATION:",
        f"    Action Plan                    : {report.clinical_recommendation}",
        border,
    ]
    return "\n".join(out)
