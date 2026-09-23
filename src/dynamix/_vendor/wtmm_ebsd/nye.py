"""Pantleon Nye-tensor assembly with grain-boundary-aware finite differences.

Replaces v1 cell 45's inlined kappa/alpha build.

Three layers:
  Stage A  -- gb_aware_grad_np (pure-NumPy stencil)
  Stage B  -- gb_aware_grad_nb (Numba @njit, parallel over rows)
  Wrapper  -- gb_aware_grad     (default Numba if available, falls back to NumPy)

Pantleon Eq. 11 & 13 (zero elastic strain, 2D-accessible):
  alpha_12 = kappa_21 = d g_2 / d x
  alpha_13 = kappa_31 = d g_3 / d x
  alpha_21 = kappa_12 = d g_1 / d y
  alpha_23 = kappa_32 = d g_3 / d y
  alpha_33 = -(kappa_11 + kappa_22)
  alpha_11 ~= -kappa_22                   (kappa_33 unmeasured -> 0)
  alpha_22 ~= -kappa_11
  alpha_31 = alpha_32 = 0                 (kappa_i3 unmeasured)
  (alpha_11 - alpha_22) = kappa_11 - kappa_22  (exact, no kappa_33 needed)

Frame: takes the SAMPLE-frame demeaned orientation field (combined_log_sample).
Mixed crystal/sample frame produces a frame-incoherent Nye tensor; demean.py
provides the rotated copy for exactly this purpose.

Public API
----------
gb_aware_grad(field, gid_map, axis, *, use_numba=True)
compute_kappa_2x3(combined_log_sample, gid_map, *, gaussian_sigma_px=0.0,
                  gb_safe=True, use_numba=True)
assemble_alpha_pantleon(kappa_2x3) -> (ny, nx, 3, 3) float64
audit_alpha_index_convention(alpha_a, alpha_b, *, mask=None, threshold=0.5)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Numba is optional; if missing we fall back to NumPy
try:
    import numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# =========================================================================
# Stage A -- pure NumPy gb-aware gradient
# =========================================================================

def gb_aware_grad_np(field: np.ndarray, gid_map: np.ndarray, axis: int) -> np.ndarray:
    """Grain-aware finite difference along `axis`. Pure NumPy.

    Stencil:
      central       if both `axis`-neighbors are in same grain
      one-sided     if exactly one `axis`-neighbor is in same grain
      NaN           if neither `axis`-neighbor is in same grain
                    or if pixel itself is unindexed (gid_map == 0)

    Parameters
    ----------
    field : (ny, nx) float
        Scalar field; for vector fields (e.g. combined_log shape (ny, nx, 3))
        call this once per component or use `compute_kappa_2x3`.
    gid_map : (ny, nx) int   0 = unindexed.
    axis : int   0 = d/dy, 1 = d/dx

    Returns
    -------
    out : (ny, nx) float64   NaN at unsafe pixels.
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"gb_aware_grad_np expects 2D field, got {f.ndim}D")
    same = gid_map > 0
    ny, nx = same.shape
    out = np.full((ny, nx), np.nan, dtype=np.float64)

    if axis == 1:
        # right-neighbor in-grain match: shape (ny, nx-1)
        same_lr = (gid_map[:, :-1] == gid_map[:, 1:]) & same[:, :-1] & same[:, 1:]
        # central available: pixel x has same_lr at x-1 AND x
        cen_ok = np.zeros_like(same)
        cen_ok[:, 1:-1] = same_lr[:, :-1] & same_lr[:, 1:]
        # central values
        cen = np.zeros_like(f)
        cen[:, 1:-1] = (f[:, 2:] - f[:, :-2]) * 0.5
        # forward-only: same_lr[x] holds (right neighbor in grain) but cen
        # is not available. Pixel index = x in [0..nx-1]; same_lr index = x.
        fwd_avail = np.zeros_like(same)
        fwd_avail[:, :-1] = same_lr
        fwd = np.zeros_like(f)
        fwd[:, :-1] = f[:, 1:] - f[:, :-1]
        # backward-only: same_lr[x-1] holds (left neighbor in grain) AND cen
        # not available AND fwd-only also not picked
        bwd_avail = np.zeros_like(same)
        bwd_avail[:, 1:] = same_lr
        bwd = np.zeros_like(f)
        bwd[:, 1:] = f[:, 1:] - f[:, :-1]

        # Apply in priority order: central > forward > backward > NaN
        out = np.where(cen_ok, cen, out)
        # forward where central failed but fwd is available
        fwd_only = fwd_avail & ~cen_ok
        out = np.where(fwd_only, fwd, out)
        # backward where neither central nor forward applied
        bwd_only = bwd_avail & ~cen_ok & ~fwd_only
        out = np.where(bwd_only, bwd, out)
        # invalid pixel (unindexed): force NaN
        out = np.where(same, out, np.nan)
        return out

    elif axis == 0:
        same_ud = (gid_map[:-1, :] == gid_map[1:, :]) & same[:-1, :] & same[1:, :]
        cen_ok = np.zeros_like(same)
        cen_ok[1:-1, :] = same_ud[:-1, :] & same_ud[1:, :]
        cen = np.zeros_like(f)
        cen[1:-1, :] = (f[2:, :] - f[:-2, :]) * 0.5
        fwd_avail = np.zeros_like(same)
        fwd_avail[:-1, :] = same_ud
        fwd = np.zeros_like(f)
        fwd[:-1, :] = f[1:, :] - f[:-1, :]
        bwd_avail = np.zeros_like(same)
        bwd_avail[1:, :] = same_ud
        bwd = np.zeros_like(f)
        bwd[1:, :] = f[1:, :] - f[:-1, :]

        out = np.where(cen_ok, cen, out)
        fwd_only = fwd_avail & ~cen_ok
        out = np.where(fwd_only, fwd, out)
        bwd_only = bwd_avail & ~cen_ok & ~fwd_only
        out = np.where(bwd_only, bwd, out)
        out = np.where(same, out, np.nan)
        return out

    else:
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")


# =========================================================================
# Stage B -- Numba-accelerated version (parallel over rows)
# =========================================================================

if _HAVE_NUMBA:

    @numba.njit(parallel=True, cache=True, fastmath=False)
    def _gb_aware_grad_kernel_axis1(field, gid_map, ny, nx, out):
        for y in numba.prange(ny):
            for x in range(nx):
                gid = gid_map[y, x]
                if gid == 0:
                    out[y, x] = np.nan
                    continue
                left_in  = (x > 0)        and gid_map[y, x - 1] == gid
                right_in = (x < nx - 1)   and gid_map[y, x + 1] == gid
                if left_in and right_in:
                    out[y, x] = 0.5 * (field[y, x + 1] - field[y, x - 1])
                elif right_in:
                    out[y, x] = field[y, x + 1] - field[y, x]
                elif left_in:
                    out[y, x] = field[y, x] - field[y, x - 1]
                else:
                    out[y, x] = np.nan

    @numba.njit(parallel=True, cache=True, fastmath=False)
    def _gb_aware_grad_kernel_axis0(field, gid_map, ny, nx, out):
        for y in numba.prange(ny):
            for x in range(nx):
                gid = gid_map[y, x]
                if gid == 0:
                    out[y, x] = np.nan
                    continue
                up_in   = (y > 0)        and gid_map[y - 1, x] == gid
                down_in = (y < ny - 1)   and gid_map[y + 1, x] == gid
                if up_in and down_in:
                    out[y, x] = 0.5 * (field[y + 1, x] - field[y - 1, x])
                elif down_in:
                    out[y, x] = field[y + 1, x] - field[y, x]
                elif up_in:
                    out[y, x] = field[y, x] - field[y - 1, x]
                else:
                    out[y, x] = np.nan

    def gb_aware_grad_nb(field: np.ndarray, gid_map: np.ndarray, axis: int) -> np.ndarray:
        """Grain-aware FD via Numba @njit(parallel=True). Same semantics as
        gb_aware_grad_np."""
        f = np.ascontiguousarray(field, dtype=np.float64)
        if f.ndim != 2:
            raise ValueError(f"gb_aware_grad_nb expects 2D field, got {f.ndim}D")
        gm = np.ascontiguousarray(gid_map, dtype=np.int64)
        ny, nx = gm.shape
        out = np.empty((ny, nx), dtype=np.float64)
        if axis == 1:
            _gb_aware_grad_kernel_axis1(f, gm, ny, nx, out)
        elif axis == 0:
            _gb_aware_grad_kernel_axis0(f, gm, ny, nx, out)
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis!r}")
        return out

else:

    def gb_aware_grad_nb(field, gid_map, axis):
        raise ImportError(
            "Numba is not installed; gb_aware_grad_nb is unavailable. "
            "Install numba>=0.59 or call gb_aware_grad_np directly."
        )


# =========================================================================
# Public dispatch wrapper
# =========================================================================

def gb_aware_grad(
    field: np.ndarray,
    gid_map: np.ndarray,
    axis: int,
    *,
    use_numba: bool = True,
) -> np.ndarray:
    """Grain-aware finite difference. Defaults to Numba if available.

    Stencil:
      central       both axis-neighbors in same grain
      one-sided     exactly one axis-neighbor in same grain
      NaN           neither, or pixel itself unindexed

    Parameters
    ----------
    field : (ny, nx) float
    gid_map : (ny, nx) int   0 = unindexed
    axis : int   0 = d/dy, 1 = d/dx
    use_numba : bool         force NumPy by passing False

    Returns
    -------
    out : (ny, nx) float64   NaN at unsafe pixels.
    """
    if use_numba and _HAVE_NUMBA:
        return gb_aware_grad_nb(field, gid_map, axis)
    return gb_aware_grad_np(field, gid_map, axis)


# =========================================================================
# Pantleon kappa + alpha assembly
# =========================================================================

def compute_kappa_2x3(
    combined_log_sample: np.ndarray,
    gid_map: np.ndarray,
    *,
    gaussian_sigma_px: float = 0.0,
    gb_safe: bool = True,
    partial_safe: bool = True,
    use_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 2D-accessible Nye-tensor curvature kappa_ij.

    kappa_2x3[..., j_spatial(0=x, 1=y), i_rotation(0=g1, 1=g2, 2=g3)]
    = d g_i / d x_j

    Parameters
    ----------
    combined_log_sample : (ny, nx, 3) float
        Sample-frame per-grain demeaned orientation field (from
        wtmm_ebsd.demean).
    gid_map : (ny, nx) int
    gaussian_sigma_px : float
        Optional Gaussian pre-smooth (spatial axes only) before FD.
        v1 default is 1.5 px for the NNLS path.  Set 0 to disable.
    gb_safe : bool
        If True, use grain-aware FD stencil; otherwise plain np.gradient.
    partial_safe : bool
        Only used when ``gb_safe=True``.  Default True.

        - True (RECOMMENDED): a pixel is "safe" if the x-axis FD OR the
          y-axis FD is finite (per-axis safety).  Components whose FD
          gave NaN are zeroed individually -- e.g. a 1-pixel-wide strip
          along x keeps its x-axis kappa components and only its y-axis
          components are zeroed.  Recovers thin-grain pixels at the cost
          of partial information per pixel.
        - False (STRICT): a pixel is "safe" only if BOTH axes' FDs are
          finite.  Whole pixel is rejected if EITHER axis fails.  This
          is the v1 behaviour and gives very low pixel coverage on scans
          with many thin grains (1-2 px wide strips).
    use_numba : bool

    Returns
    -------
    kappa_2x3 : (ny, nx, 2, 3) float64
        NaN components are replaced with 0 in the output.
    gb_unsafe : (ny, nx) bool
        True where the pixel contributes NOTHING to kappa.  With
        partial_safe=True this is True only where BOTH axes failed AND
        the pixel is unindexed; with partial_safe=False it's True if
        ANY axis failed (the v1 stricter definition).
    """
    if combined_log_sample.ndim != 3 or combined_log_sample.shape[-1] != 3:
        raise ValueError(
            f"combined_log_sample must be (ny, nx, 3), got {combined_log_sample.shape}"
        )
    src = np.asarray(combined_log_sample, dtype=np.float64)

    if gaussian_sigma_px and gaussian_sigma_px > 0:
        from scipy.ndimage import gaussian_filter
        src = gaussian_filter(src, sigma=[gaussian_sigma_px, gaussian_sigma_px, 0.0])

    ny, nx, _ = src.shape
    kappa = np.zeros((ny, nx, 2, 3), dtype=np.float64)

    if gb_safe:
        for ci in range(3):
            kappa[..., 0, ci] = gb_aware_grad(src[..., ci], gid_map, axis=1,
                                              use_numba=use_numba)
            kappa[..., 1, ci] = gb_aware_grad(src[..., ci], gid_map, axis=0,
                                              use_numba=use_numba)
        # Per-axis safety: a pixel's x-axis FD is OK if ALL three components
        # came back finite for axis 1 (since the same gid neighbour test
        # is shared across the 3 components, this is identical to checking
        # one component).  Same for y-axis.
        x_axis_ok = np.isfinite(kappa[..., 0, :]).all(axis=-1)
        y_axis_ok = np.isfinite(kappa[..., 1, :]).all(axis=-1)
        if partial_safe:
            # Pixel contributes if AT LEAST one axis is OK and pixel is indexed
            gb_unsafe = (~(x_axis_ok | y_axis_ok)) | (gid_map == 0)
        else:
            # Strict: pixel contributes only if BOTH axes are OK
            gb_unsafe = (~(x_axis_ok & y_axis_ok)) | (gid_map == 0)
        # Zero out NaNs at the COMPONENT level so a pixel with just y-axis
        # safe still contributes its (still-valid) y-axis kappa values.
        kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        for ci in range(3):
            kappa[..., 0, ci] = np.gradient(src[..., ci], axis=1)
            kappa[..., 1, ci] = np.gradient(src[..., ci], axis=0)
        gb_unsafe = np.zeros((ny, nx), dtype=bool)

    return kappa, gb_unsafe


def assemble_alpha_pantleon(kappa_2x3: np.ndarray) -> np.ndarray:
    """Build the Nye tensor alpha (Pantleon 2008 Eqs. 11 & 13).

    Off-diagonals (i != k):  alpha_ik = kappa_ki
        alpha_12 = kappa_21         alpha_21 = kappa_12
        alpha_13 = kappa_31         alpha_23 = kappa_32
        alpha_31 = 0                alpha_32 = 0   (kappa_i3 unmeasured)

    Diagonals:
        alpha_33 = -(kappa_11 + kappa_22)               (exact)
        alpha_11 = -kappa_22                            (kappa_33 ~= 0)
        alpha_22 = -kappa_11                            (kappa_33 ~= 0)
        (alpha_11 - alpha_22) = kappa_11 - kappa_22     (exact)

    Array layout: kappa_2x3[..., j_spatial(0=x,1=y), i_rotation(0=g1,1=g2,2=g3)]
                  alpha_3x3[..., i_rotation, j_spatial]   shape (..., 3, 3)

    The output's third spatial column (j=2) is zero because kappa_i3 is
    unmeasured in 2D EBSD.

    Parameters
    ----------
    kappa_2x3 : (ny, nx, 2, 3) float

    Returns
    -------
    alpha_3x3 : (ny, nx, 3, 3) float64
    """
    if kappa_2x3.shape[-2:] != (2, 3):
        raise ValueError(
            f"kappa_2x3 must end in (2, 3), got {kappa_2x3.shape[-2:]}"
        )
    k = np.asarray(kappa_2x3, dtype=np.float64)
    ny, nx = k.shape[:2]

    k11 = k[..., 0, 0]    # d g_1 / d x
    k21 = k[..., 0, 1]    # d g_2 / d x
    k31 = k[..., 0, 2]    # d g_3 / d x
    k12 = k[..., 1, 0]    # d g_1 / d y
    k22 = k[..., 1, 1]    # d g_2 / d y
    k32 = k[..., 1, 2]    # d g_3 / d y

    alpha = np.zeros((ny, nx, 3, 3), dtype=np.float64)
    alpha[..., 0, 1] = k21    # alpha_12 = kappa_21
    alpha[..., 0, 2] = k31    # alpha_13 = kappa_31
    alpha[..., 1, 0] = k12    # alpha_21 = kappa_12
    alpha[..., 1, 2] = k32    # alpha_23 = kappa_32
    alpha[..., 2, 0] = 0.0    # alpha_31 unmeasured
    alpha[..., 2, 1] = 0.0    # alpha_32 unmeasured
    alpha[..., 2, 2] = -(k11 + k22)
    alpha[..., 0, 0] = -k22
    alpha[..., 1, 1] = -k11
    return alpha


# =========================================================================
# Index-convention audit
# =========================================================================

def audit_alpha_index_convention(
    alpha_a: dict,
    alpha_b: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Verify two alpha sources agree on the Pantleon Eq. 11/13 component
    mapping.  Used to guard against the v1-style index transposition that
    silently put alpha_12 in the alpha_21 slot.

    Parameters
    ----------
    alpha_a : dict[str, ndarray]
        Per-component scalar alpha fields, keyed by Pantleon name (e.g.
        'a12', 'a13', 'a21', 'a23', 'a33').  Each value is (ny, nx) float.
    alpha_b : (ny, nx, 3, 3) ndarray
        Tensor alpha (output of assemble_alpha_pantleon).  Indexed
        [..., i_rotation, j_spatial] (1-indexed Pantleon -> 0-indexed: -1).
    mask : (ny, nx) bool, optional
        Restrict the comparison to these pixels (e.g. interior of grains).
    threshold : float
        Pearson r below this signals an index-convention mismatch.

    Returns
    -------
    audit : dict[str, float]    component name -> Pearson r.  Negative or
                                near-zero values indicate a transposition.
    """
    name_to_idx = {
        'a12': (0, 1), 'a13': (0, 2),
        'a21': (1, 0), 'a23': (1, 2),
        'a33': (2, 2),
    }
    out = {}
    for name, fld_a in alpha_a.items():
        if name not in name_to_idx:
            continue
        i, j = name_to_idx[name]
        fld_b = alpha_b[..., i, j]
        if mask is not None:
            a = np.asarray(fld_a)[mask].astype(np.float64).ravel()
            b = np.asarray(fld_b)[mask].astype(np.float64).ravel()
        else:
            a = np.asarray(fld_a).astype(np.float64).ravel()
            b = np.asarray(fld_b).astype(np.float64).ravel()
        if a.std() == 0 or b.std() == 0:
            out[name] = float('nan')
        else:
            out[name] = float(np.corrcoef(a, b)[0, 1])
    return out
