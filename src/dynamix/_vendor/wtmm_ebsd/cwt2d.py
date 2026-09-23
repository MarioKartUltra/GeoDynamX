"""2D CWT for EBSD orientation/scalar fields, MLX-accelerated.

Lifted from v1 cell 2 (definitions inline'd in every EBSD notebook).
Identical numerical behavior; just packaged.

The CWT operates on 2D scalar fields and returns wavelet derivatives at
each scale.  The "wavelet" parameter selects the SMOOTHING function in
the xsmurf convention -- the actual ANALYZING wavelet is the smoothing
function differentiated by the chosen `derivs` order
(see wtmm.wavelet_scalebar).

Public API
----------
compute_scales(n_oct, n_vox, a_min=1)        -> list of scales
cwt_2d_f32(image, scales, *, pad=32,
           derivs='first', wavelet='gaussian',
           verbose=True)                      -> dict of (n_sc, ny, nx) arrays
tensor_svd_3x2(gx1, gy1, gx2, gy2, gx3, gy3)  -> (sigma_max, sigma_min, arg)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.fft import fftfreq

# MLX (Apple GPU) is required for the float32 fast path.  No NumPy fallback
# is provided here; the same constraint applies to the v1 notebook.
try:                                   # GeoDynamix_Beta: Windows has no mlx -- the
    import mlx.core as mx              # app then runs this transform on FFTW3
except ImportError:                    # (dynamix.core.fft_policy); cwt_2d_f32 needs mx.
    mx = None


def compute_scales(n_oct: int, n_vox: int, a_min: float = 1.0) -> list:
    """Logarithmically spaced scales (xsmurf convention).

    norm = 6.0 / 0.86; scales[k] = a_min * 2 ** (oct + voice/n_vox) * norm.
    Returns a Python list (not ndarray) for compatibility with v1 API.
    """
    norm = 6.0 / 0.86
    return [a_min * (2.0 ** (o + v / n_vox)) * norm
            for o in range(n_oct) for v in range(n_vox)]


def _build_wavelet_filters_f32(kx, ky, scale: float, wavelet: str = 'gaussian'):
    """Build Fourier-domain derivative filters at one scale.

    `wavelet` is the SMOOTHING function name: 'gaussian' (default) or
    'mexican' (which uses |k|^2 * gaussian as the smoother).  The actual
    derivative shape is set by the keys of the returned dict.
    """
    sx = kx * scale
    sy = ky * scale
    gauss = mx.exp(-(sx * sx + sy * sy))
    zero = mx.zeros_like(gauss)
    def to_complex(r, i):
        return mx.stack([r, i], axis=-1).view(mx.complex64).squeeze(-1)
    if wavelet == 'mexican':
        k2 = sx * sx + sy * sy
        return {
            'dx':   to_complex(zero,                 sx * k2 * gauss),
            'dy':   to_complex(zero,                 sy * k2 * gauss),
            'dxx':  to_complex(-sx * sx * k2 * gauss, zero),
            'dxy':  to_complex(-sy * sx * k2 * gauss, zero),
            'dyy':  to_complex(-sy * sy * k2 * gauss, zero),
            'dxxx': to_complex(zero, -sx * sx * sx * k2 * gauss),
            'dxxy': to_complex(zero, -sx * sx * sy * k2 * gauss),
            'dxyy': to_complex(zero, -sx * sy * sy * k2 * gauss),
            'dyyy': to_complex(zero, -sy * sy * sy * k2 * gauss),
        }
    else:
        return {
            'dx':   to_complex(zero,                 sx * gauss),
            'dy':   to_complex(zero,                 sy * gauss),
            'dxx':  to_complex(-sx * sx * gauss,     zero),
            'dxy':  to_complex(-sy * sx * gauss,     zero),
            'dyy':  to_complex(-sy * sy * gauss,     zero),
            'dxxx': to_complex(zero, -sx * sx * sx * gauss),
            'dxxy': to_complex(zero, -sx * sx * sy * gauss),
            'dxyy': to_complex(zero, -sx * sy * sy * gauss),
            'dyyy': to_complex(zero, -sy * sy * sy * gauss),
        }


def cwt_2d_f32(
    image: np.ndarray,
    scales,
    *,
    pad: int = 32,
    derivs: str = 'first',
    wavelet: str = 'gaussian',
    verbose: bool = True,
) -> dict:
    """2D CWT via MLX FFT, float32.

    Parameters
    ----------
    image : (ny, nx) float       2D scalar field.
    scales : iterable of float   wavelet scales in pixels.
    pad : int                    mirror-pad width in pixels (default 32).
    derivs : str
        'first' -> only dx, dy returned (plus mod, arg).
        'all'   -> up to third-order derivatives (dx, dy, dxx, dxy, dyy,
                   dxxx, dxxy, dxyy, dyyy).
    wavelet : str                'gaussian' (default) or 'mexican'.
    verbose : bool

    Returns
    -------
    dict
        Keys: requested derivative names + 'mod' + 'arg'.
        Each value is (n_scales, ny, nx) float32.
    """
    ny, nx = image.shape
    padded = np.pad(image, pad, mode='reflect').astype(np.float32)
    ny_p, nx_p = padded.shape
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    kx_np = fftfreq(nx_p).astype(np.float32)
    ky_np = fftfreq(ny_p).astype(np.float32)
    KX_np, KY_np = np.meshgrid(kx_np, ky_np)
    KX, KY = mx.array(KX_np), mx.array(KY_np)
    image_fft = mx.fft.fft2(mx.array(padded))
    if derivs == 'first':
        names = ['dx', 'dy']
    else:
        names = ['dx', 'dy', 'dxx', 'dxy', 'dyy',
                 'dxxx', 'dxxy', 'dxyy', 'dyyy']
    n_scales = len(scales)
    result = {n: np.zeros((n_scales, ny, nx), dtype=np.float32) for n in names}
    result['mod'] = np.zeros((n_scales, ny, nx), dtype=np.float32)
    result['arg'] = np.zeros((n_scales, ny, nx), dtype=np.float32)
    for i, scale in enumerate(scales):
        filters = _build_wavelet_filters_f32(KX, KY, scale, wavelet=wavelet)
        for name in names:
            conv = mx.fft.ifft2(image_fft * filters[name])
            result[name][i] = np.array(conv.real)[crop]
        dx_s, dy_s = result['dx'][i], result['dy'][i]
        result['mod'][i] = np.sqrt(dx_s * dx_s + dy_s * dy_s)
        result['arg'][i] = np.arctan2(dy_s, dx_s).astype(np.float32)
        if verbose and (i % max(1, n_scales // 4) == 0 or i == n_scales - 1):
            print(f'  Scale {i}/{n_scales-1}: a={scale:.1f}')
    mx.eval()
    return result


def tensor_svd_3x2(
    gx1, gy1, gx2, gy2, gx3, gy3,
):
    """SVD of the 3x2 Jacobian (g_i_j with i in {1,2,3}, j in {x,y}) at every
    pixel via closed-form J^T J eigendecomposition.

    All inputs are (ny, nx) float arrays.  Returns (sigma_max, sigma_min, arg)
    where arg is the orientation of the leading right singular vector (rad).
    """
    a = gx1 * gx1 + gx2 * gx2 + gx3 * gx3
    b = gx1 * gy1 + gx2 * gy2 + gx3 * gy3
    d = gy1 * gy1 + gy2 * gy2 + gy3 * gy3
    trace = a + d
    det = a * d - b * b
    disc = np.sqrt(np.maximum((trace / np.float32(2)) ** 2 - det,
                                np.float32(0)))
    lam1 = trace / np.float32(2) + disc
    lam2 = np.maximum(trace / np.float32(2) - disc, np.float32(0))
    sigma_max = np.sqrt(np.maximum(lam1, np.float32(0))).astype(np.float32)
    sigma_min = np.sqrt(lam2).astype(np.float32)
    vx = lam1 - d
    vy = b
    vnorm = np.sqrt(vx ** 2 + vy ** 2) + np.float32(1e-30)
    arg = np.arctan2(vy / vnorm, vx / vnorm).astype(np.float32)
    return sigma_max, sigma_min, arg
