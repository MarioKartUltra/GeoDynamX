"""wtmm_ebsd -- EBSD-specific extensions to the wtmm multifractal toolkit.

Sister package to `wtmm` and `xsmurf_wrapper`. Provides:
  - orientation algebra (quaternion ops, Karcher mean, frame rotations)
  - symmetry handling (Laue groups, FZ thresholds, Dauphine awareness)
  - grain segmentation (flood-fill, smart fill, GB masks, distance-to-GB)
  - per-grain demean (combined_log + sample-frame variant)
  - Nye tensor assembly (Pantleon Eqs. 11/13, GB-aware finite differences)
  - KAM (orix-based) + Lebyodkin et al. 2024 box-counting MF
  - CPO attractor + directional anisotropy
  - Papeschi 2019 quartz-specific diagnostics (Dauphine, MAD, MOCC, MOSC)
  - alpha-Jacobian TWTMM driver
  - chain pruning (modulus & Holder-slope thresholds)

Designed to be consumed from EBSD analysis notebooks; widgets stay in the
notebook. See PERGRAIN_PANTLEON_REFACTOR_PLAN.md in the repo root.
"""

__version__ = "0.0.1.dev0"

# GeoDynamix_Beta vendoring (2026-09-22): the EBSD modules (orientation, symmetry, grains,
# demean, cpo, io, ipf, kam, papeschi, orientation_stats, scalar_fields, mtex_bridge) are LEFT
# OUT -- no orix, no MATLAB, no EBSD reading in the beta. Only these modules are imported.

# ---- public re-exports -----------------------------------------------------
from . import (
    chain_filters,
    cwt2d,
    nye,
    partition,
    quality,
    shuffles,
    spatial_stats,
    twtmm,
    viewer,
)

# Most-used names exposed at package level for ergonomics in notebooks.
from .nye import (
    gb_aware_grad,
    gb_aware_grad_np,
    compute_kappa_2x3,
    assemble_alpha_pantleon,
    audit_alpha_index_convention,
)
from .cwt2d import (
    compute_scales,
    cwt_2d_f32,
    tensor_svd_3x2,
)
from .quality import (
    bc_gradient_magnitude,
    bc_damage_mask,
    confidence_weighted_damage,
)
from .twtmm import (
    alpha_jacobian_twtmm,
    scalar_field_wtmm,
)
from .viewer import (
    precompute_viewer_bundle,
    viewer_bundle_diagnostics,
    add_wavelet_scalebar,
    add_map_scalebar,
)
from .partition import (
    DEFAULT_Q_LIST,
    compute_cumulants_log2,
    build_hd_from_chains,
    fit_hq_Dq_weighted,
    dupont_fit_polynomial,
    venugopal_fit_cumulants,
    tau_from_cumulants,
    Dh_parabola_from_cumulants,
)
