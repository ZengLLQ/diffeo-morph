"""Clinical Alzheimer's Disease Neurodegeneration & Atrophy Biomarker Assessment."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from diffeo_morph.clinical.morphometry import RegionalMorphometryMetrics


@dataclass
class AlzheimerClinicalReport:
    """Clinical diagnostic report for longitudinal neurodegeneration."""
    hippocampal_change_pct: float
    ventricular_change_pct: float
    annualized_hippocampal_atrophy_rate: float
    clinical_stage: Literal[
        "NORMAL_AGING",
        "MILD_COGNITIVE_IMPAIRMENT",
        "PROBABLE_ALZHEIMERS_DISEASE",
        "RAPID_NEURODEGENERATION",
    ]
    confidence_score: float
    diagnostic_summary: str
    clinical_recommendation: str


class AlzheimerEvaluator:
    """
    Evaluates annualized atrophy metrics against clinical neuroimaging thresholds
    derived from the Alzheimer's Disease Neuroimaging Initiative (ADNI).
    """

    @staticmethod
    def evaluate_longitudinal_scan(
        hippocampus_metrics: RegionalMorphometryMetrics,
        ventricle_metrics: RegionalMorphometryMetrics,
        scan_interval_months: float = 12.0,
    ) -> AlzheimerClinicalReport:
        """
        Assess neurodegenerative progression from longitudinal MRI pair.
        """
        annual_factor = 12.0 / max(1.0, scan_interval_months)
        annual_hippo_atrophy = -hippocampus_metrics.percent_volume_change * annual_factor
        annual_ventricle_expansion = ventricle_metrics.percent_volume_change * annual_factor

        # Clinical ADNI criteria:
        # Normal aging: Hippocampal loss < 1.5% / year, Ventricular growth < 3.0% / year
        # MCI: Hippocampal loss 1.5% - 4.0% / year, Ventricular growth 3.0% - 6.0% / year
        # Probable AD: Hippocampal loss 4.0% - 8.0% / year, Ventricular growth > 6.0% / year
        # Rapid: Hippocampal loss > 8.0% / year

        if annual_hippo_atrophy > 8.0:
            stage = "RAPID_NEURODEGENERATION"
            confidence = 0.95
            summary = (
                f"Severe accelerated hippocampal atrophy ({annual_hippo_atrophy:.1f}%/yr) "
                f"with marked ventricular ex-vacuo expansion ({annual_ventricle_expansion:.1f}%/yr)."
            )
            recom = "Urgent memory clinic evaluation, CSF/PET amyloid biomarker confirmation, and clinical trial enrollment."
        elif annual_hippo_atrophy > 4.0 or annual_ventricle_expansion > 6.0:
            stage = "PROBABLE_ALZHEIMERS_DISEASE"
            confidence = 0.88
            summary = (
                f"Clinically significant hippocampal volume loss ({annual_hippo_atrophy:.1f}%/yr) "
                f"consistent with typical Alzheimer's disease pathology."
            )
            recom = "Initiate disease-modifying anti-amyloid therapy consultation and cognitive monitoring."
        elif annual_hippo_atrophy > 1.8 or annual_ventricle_expansion > 3.5:
            stage = "MILD_COGNITIVE_IMPAIRMENT"
            confidence = 0.80
            summary = (
                f"Moderate hippocampal atrophy ({annual_hippo_atrophy:.1f}%/yr) exceeding normal age-adjusted decline; "
                f"suggestive of prodromal amnestic MCI."
            )
            recom = "Follow-up MRI in 6-12 months with MoCA/MMSE neuropsychological tracking."
        else:
            stage = "NORMAL_AGING"
            confidence = 0.92
            summary = (
                f"Hippocampal volume preserved ({annual_hippo_atrophy:.1f}%/yr decline within physiological limits). "
                f"No evidence of pathological neurodegeneration."
            )
            recom = "Routine lifestyle maintenance and age-appropriate follow-up."

        return AlzheimerClinicalReport(
            hippocampal_change_pct=hippocampus_metrics.percent_volume_change,
            ventricular_change_pct=ventricle_metrics.percent_volume_change,
            annualized_hippocampal_atrophy_rate=annual_hippo_atrophy,
            clinical_stage=stage,
            confidence_score=confidence,
            diagnostic_summary=summary,
            clinical_recommendation=recom,
        )
