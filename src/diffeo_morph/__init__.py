"""DiffeoMorph: LDDMM & Lie Group SVF 3D Brain MRI Morphometry Engine."""

from diffeo_morph.geometry import (
    create_identity_grid,
    warp_image,
    compose_transformations,
    scaling_and_squaring,
    compute_jacobian_matrix,
    compute_jacobian_determinant,
    count_foldings,
    compute_log_jacobian,
)
from diffeo_morph.registration import (
    mean_squared_error,
    local_normalized_cross_correlation,
    VectorRegularizer,
    RegistrationResult,
    SVFRegistrationOptimizer,
)
from diffeo_morph.clinical import (
    RegionalMorphometryMetrics,
    TensorBasedMorphometry,
    AlzheimerClinicalReport,
    AlzheimerEvaluator,
)

__version__ = "0.1.0"

__all__ = [
    "create_identity_grid",
    "warp_image",
    "compose_transformations",
    "scaling_and_squaring",
    "compute_jacobian_matrix",
    "compute_jacobian_determinant",
    "count_foldings",
    "compute_log_jacobian",
    "mean_squared_error",
    "local_normalized_cross_correlation",
    "VectorRegularizer",
    "RegistrationResult",
    "SVFRegistrationOptimizer",
    "RegionalMorphometryMetrics",
    "TensorBasedMorphometry",
    "AlzheimerClinicalReport",
    "AlzheimerEvaluator",
]
