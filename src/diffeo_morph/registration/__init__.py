"""Image registration metrics, regularizers, and SVF optimization engine."""

from diffeo_morph.registration.metrics import (
    mean_squared_error,
    local_normalized_cross_correlation,
)
from diffeo_morph.registration.regularizers import VectorRegularizer
from diffeo_morph.registration.svf_optimizer import (
    RegistrationResult,
    SVFRegistrationOptimizer,
)

__all__ = [
    "mean_squared_error",
    "local_normalized_cross_correlation",
    "VectorRegularizer",
    "RegistrationResult",
    "SVFRegistrationOptimizer",
]
