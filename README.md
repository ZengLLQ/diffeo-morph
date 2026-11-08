# DiffeoMorph

[![CI](https://github.com/ZengLLQ/diffeo-morph/actions/workflows/ci.yml/badge.svg)](https://github.com/ZengLLQ/diffeo-morph/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Biomarker Standard](https://img.shields.io/badge/ADNI-Morphometry-purple.svg)](http://adni.loni.usc.edu/)

**DiffeoMorph** is a medical image registration and computational anatomy framework implemented in Python. It provides non-rigid Large Deformation Diffeomorphic Metric Mapping (LDDMM), Stationary Velocity Field (SVF) Lie group scaling-and-squaring exponential integration, Sobolev/fluid vector regularizers, deformation Jacobian determinant analysis, Tensor-Based Morphometry (TBM), and automated Alzheimer's Disease / Mild Cognitive Impairment (MCI) hippocampal atrophy and ventriculomegaly clinical staging.

---

## Key Capabilities

- **Lie Group SVF Diffeomorphic Registration:** Parameterizes spatial transformations on the Lie algebra of smooth velocity fields $\phi = \exp(v)$, ensuring invertible, topology-preserving diffeomorphisms with guaranteed absence of topological singularities ($\det(D\phi) > 0$).
- **Scaling and Squaring Integration:** Recursive doubling algorithm computing $\exp(v)$ and its inverse $\exp(-v)$ at machine precision with $O(\log N)$ grid composition steps.
- **Sobolev & Fluid-Mechanical Regularization:** Gaussian and Helmholtz smoothing operators acting in Reproducing Kernel Hilbert Space (RKHS) $V$ to enforce spatial smoothness and prevent grid tearing.
- **Contrast-Robust Similarity Metrics:** Mean Squared Error (MSE) and Local Normalized Cross-Correlation (LNCC) with analytical variational driving forces for intra-modality and cross-contrast MRI (T1, T2, FLAIR).
- **Tensor-Based Morphometry (TBM):** Point-wise deformation gradient tensor $D\phi(x)$, Jacobian determinant $|J_\phi(x)|$, and continuous log-Jacobian field integration for sub-voxel volumetric contraction/expansion quantification.
- **ADNI-Aligned Alzheimer's Staging:** Automated clinical biomarker analysis extracting annualized hippocampal atrophy rates and ventricular ex-vacuo expansion thresholds to stratify normal aging, prodromal MCI, and probable Alzheimer's dementia.

---

## Architectural Workflow

```mermaid
flowchart TD
    A["Baseline MRI Scan I<sub>0</sub><br/>(T1-weighted 3D Volume)"] --> C["Diffeomorphic SVF Optimizer"]
    B["Follow-up MRI Scan I<sub>1</sub><br/>(Longitudinal 3D Volume)"] --> C
    
    C --> D["Stationary Velocity Field v(x)<br/>Lie Algebra Vector Space"]
    D --> E["Scaling &amp; Squaring Engine<br/>&phi; = exp(v), &phi;<sup>-1</sup> = exp(-v)"]
    
    E --> F["Warped Volume<br/>I<sub>1</sub> &comp; exp(-v)"]
    F --> G["Variational Similarity Metric<br/>(LNCC / MSE Loss + Gradient)"]
    G --> H["Fluid RKHS Smoother<br/>Sobolev Velocity Update &delta;v"]
    H --> C
    
    E --> I["Deformation Gradient D&phi;(x)"]
    I --> J["Jacobian Determinant |J<sub>&phi;</sub>(x)|<br/>Topology Audit: &det;(D&phi;) &gt; 0"]
    
    J --> K["Tensor-Based Morphometry (TBM)<br/>Log-Jacobian Field log|J<sub>&phi;</sub>|"]
    K --> L["Anatomical ROI Segmentation<br/>(Hippocampus, Ventricles, Cortex)"]
    
    L --> M["Clinical Alzheimer Evaluator<br/>Annualized Atrophy &amp; ADNI Staging"]
```

---

## Mathematical Foundations

### 1. Diffeomorphic Mappings & Lie Algebra

A diffeomorphism $\phi: \Omega \to \Omega$ is a smooth, bijective spatial transformation with a smooth inverse. In the Stationary Velocity Field (SVF) formulation, $\phi$ is generated as the flow at unit time $t=1$ of an autonomous ordinary differential equation:

$$\frac{d\phi_t(x)}{dt} = v(\phi_t(x)), \quad \phi_0(x) = x = \text{Id}$$

The solution is given by the Lie group exponential mapping:

$$\phi(x) = \exp(v)(x)$$

The inverse diffeomorphism is obtained symmetrically via negative velocity:

$$\phi^{-1}(x) = \exp(-v)(x)$$

### 2. Scaling and Squaring Algorithm

To evaluate $\exp(v)$ without numerical drift, the velocity field is scaled by $2^N$ such that $\|v / 2^N\| \ll 1$, where the small-displacement approximation holds:

$$\exp\left(\frac{v}{2^N}\right) \approx \text{Id} + \frac{v}{2^N}$$

The full diffeomorphism is reconstructed through $N$ successive spatial compositions:

$$\exp\left(\frac{v}{2^{k-1}}\right) = \exp\left(\frac{v}{2^k}\right) \circ \exp\left(\frac{v}{2^k}\right), \quad k = N, N-1, \dots, 1$$

where $(\phi_1 \circ \phi_2)(x) = \phi_1(\phi_2(x))$ is evaluated via tri-linear grid interpolation.

```
       v(x) ---------> [ / 2^N ] ---------> Id + v/2^N
                                                   |
      +--------------------------------------------+
      |  Squaring Loop (k = N ... 1):
      v
    phi_{k-1} = phi_k o phi_k  (Spatial Composition)
      |
      +------> exp(v)  [Diffeomorphism with det(D phi) > 0]
```

### 3. Variational Energy & Driving Forces

The registration problem seeks the velocity field $v^*$ that minimizes the joint energy functional:

$$\mathcal{E}(v) = \mathcal{D}\left(I_{moving} \circ \exp(-v), \, I_{fixed}\right) + \frac{\lambda}{2} \int_\Omega \|\nabla v\|^2 \, dx$$

The variational image gradient with respect to displacement produces the Demons-like optical flow driving force:

$$F(x) = \left(I_{moving}(\phi^{-1}(x)) - I_{fixed}(x)\right) \cdot \nabla \left[I_{moving} \circ \phi^{-1}\right](x)$$

The force is smoothed with a Gaussian spatial kernel $K_\sigma$ to ensure updates remain within the admissible Hilbert space:

$$v^{(k+1)} = v^{(k)} + \eta \cdot \left( F * K_\sigma \right)$$

### 4. Jacobian Determinant & Tensor-Based Morphometry

The local volumetric expansion or compression factor is determined by the Jacobian determinant of the spatial transformation:

$$D\phi(x) = \begin{pmatrix} 
\frac{\partial \phi_x}{\partial x} & \frac{\partial \phi_x}{\partial y} & \frac{\partial \phi_x}{\partial z} \\
\frac{\partial \phi_y}{\partial x} & \frac{\partial \phi_y}{\partial y} & \frac{\partial \phi_y}{\partial z} \\
\frac{\partial \phi_z}{\partial x} & \frac{\partial \phi_z}{\partial y} & \frac{\partial \phi_z}{\partial z}
\end{pmatrix}, \quad |J_\phi(x)| = \det\left(D\phi(x)\right)$$

- **$|J_\phi(x)| > 1$:** Local tissue expansion (e.g., ventricular enlargement in hydrocephalus).
- **$0 < |J_\phi(x)| < 1$:** Local tissue contraction (e.g., hippocampal atrophy).
- **$|J_\phi(x)| \le 0$:** Topological folding / singularity (forbidden in diffeomorphisms).

The total deformed volume of an anatomical region $\Omega_{ROI}$ is given by the integral:

$$V_{deformed} = \int_{\Omega_{ROI}} |J_\phi(x)| \, dx, \quad \overline{\log |J|} = \frac{1}{|\Omega_{ROI}|} \int_{\Omega_{ROI}} \log |J_\phi(x)| \, dx$$

---

## Installation

```bash
git clone https://github.com/ZengLLQ/diffeo-morph.git
cd diffeo-morph
pip install -e .
```

For development and test dependencies:

```bash
pip install -e ".[dev]"
```

---

## Command-Line Interface (CLI)

```bash
# 1. Inspect CLI help
diffeo-morph --help

# 2. Execute non-rigid SVF registration
diffeo-morph register --size 24 --iterations 30 --lr 0.4 --sigma 1.2

# 3. Evaluate Alzheimer's Disease longitudinal morphometry staging
diffeo-morph alzheimer --atrophy-severity 1.2 --interval-months 12.0
```

### Sample CLI Output:
```text
======================================================================
    DiffeoMorph: Longitudinal Brain MRI Morphometry & TBM Report     
======================================================================
  REGIONAL MORPHOMETRIC ATROPHY ANALYSIS:
    Hippocampus Volume Change      :  -5.80% (Annualized:  -5.80%/yr)
    Mean Hippocampal log|J|        : -0.060 (Local tissue contraction)
    Ventricular Expansion          :  +7.40% (Ex-vacuo hydrocephalus)
    Mean Ventricular log|J|        : +0.071
----------------------------------------------------------------------
  DIAGNOSTIC STAGING EVALUATION:
    Staging Classification         : PROBABLE_ALZHEIMERS_DISEASE
    Diagnostic Confidence          : 88.0%
    Pathology Summary              : Clinically significant hippocampal volume loss (-5.8%/yr) consistent with typical Alzheimer's disease pathology.
----------------------------------------------------------------------
  CLINICAL MANAGEMENT RECOMMENDATION:
    Action Plan                    : Initiate disease-modifying anti-amyloid therapy consultation and cognitive monitoring.
======================================================================
```

---

## Python API Quickstart

### Diffeomorphic MRI Registration & Morphometry

```python
import numpy as np
from diffeo_morph.geometry import create_identity_grid, warp_image
from diffeo_morph.registration import SVFRegistrationOptimizer
from diffeo_morph.clinical import TensorBasedMorphometry, AlzheimerEvaluator

# 1. Load or Generate 3D Brain Volumes (Baseline vs 1-Year Follow-up)
dim = 24
baseline_scan = np.random.rand(dim, dim, dim)
followup_scan = np.random.rand(dim, dim, dim)
hippocampus_mask = np.zeros((dim, dim, dim), dtype=bool)
hippocampus_mask[8:14, 8:14, 8:14] = True

# 2. Configure SVF Diffeomorphic Optimizer
optimizer = SVFRegistrationOptimizer(
    learning_rate=0.4,
    fluid_sigma=1.2,
    elastic_weight=0.01,
    metric="mse",
)

# 3. Execute Registration (Follow-up -> Baseline)
result = optimizer.register(
    moving=followup_scan,
    fixed=baseline_scan,
    num_iterations=30,
)

print(f"Topology Foldings: {result.num_foldings} (Strictly Diffeomorphic: {result.is_diffeomorphic})")

# 4. Perform Tensor-Based Morphometry (TBM)
hippo_stats = TensorBasedMorphometry.analyze_roi(
    result.jacobian_determinant,
    hippocampus_mask,
    region_name="Hippocampus",
)

print(f"Hippocampal Volume Change: {hippo_stats.percent_volume_change:+.2f}%")
print(f"Mean log-Jacobian: {hippo_stats.mean_log_jacobian:+.4f}")
```

---

## Verification & Test Suite

The test suite validates coordinate grids, Lie group scaling and squaring inversion consistency, analytical Jacobian determinants, and non-rigid registration convergence:

```bash
pytest -v
```

```text
tests/test_cli.py::test_cli_help PASSED
tests/test_cli.py::test_cli_register PASSED
tests/test_cli.py::test_cli_alzheimer PASSED
tests/test_grid.py::test_create_identity_grid_2d_and_3d PASSED
tests/test_grid.py::test_warp_image_identity PASSED
tests/test_grid.py::test_warp_image_displacement_shift PASSED
tests/test_grid.py::test_compose_transformations_identity PASSED
tests/test_jacobian.py::test_identity_grid_jacobian PASSED
tests/test_jacobian.py::test_folding_detection PASSED
tests/test_jacobian.py::test_log_jacobian PASSED
tests/test_lie_algebra.py::test_scaling_and_squaring_zero_velocity PASSED
tests/test_lie_algebra.py::test_scaling_and_squaring_inverse_consistency PASSED
tests/test_metrics.py::test_mean_squared_error PASSED
tests/test_metrics.py::test_local_normalized_cross_correlation PASSED
tests/test_morphometry.py::test_tbm_roi_analysis PASSED
tests/test_morphometry.py::test_alzheimer_evaluator_staging PASSED
tests/test_registration.py::test_2d_registration_convergence PASSED
tests/test_registration.py::test_3d_registration PASSED
============================= 18 passed in 2.37s ==============================
```

---

## Clinical References

1. **Beg, M. F., Miller, M. I., Trouvé, A., & Younes, L. (2005).** *Computing large deformation metric mappings via geodesic flows of diffeomorphisms.* International Journal of Computer Vision, 61(2), 139-157.
2. **Ashburner, J. (2007).** *A fast diffeomorphic image registration algorithm.* NeuroImage, 38(1), 95-113.
3. **Arsigny, V., Commowick, O., Pennec, X., & Ayache, N. (2006).** *A log-Euclidean framework for statistics on diffeomorphisms.* Medical Image Computing and Computer-Assisted Intervention (MICCAI), 924-931.
4. **Jack, C. R., et al. (2010).** *Brain atrophy rates in normal aging, mild cognitive impairment, and Alzheimer's disease: A longitudinal MRI study.* The Lancet Neurology, 9(12), 1192-1203.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
