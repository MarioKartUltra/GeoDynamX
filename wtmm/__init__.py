"""WTMM (Wavelet Transform Modulus Maxima) 1D analysis package.

Extracted from shared notebook code for reuse across multiple backends
(torch GPU, FFTW CPU, MLX Apple Silicon, compressed sensing).

Modules with optional dependencies (numba, torch, pyfftw, mlx) use lazy
imports — they are only loaded when you import from them explicitly.
"""

# --- Colormaps (always available) ---
from .colormaps import (
    lastwave_colormap,
    lastwave_grey_colormap,
    lastwave_b2r_colormap,
    scalogram_lastwave,
)

# --- Wavelets (always available) ---
from .wavelets import WAVELETS, wavelet_direct, wavelet_support

# --- CWT engines (torch/pyfftw/mlx lazy-imported inside functions) ---
from .cwt import compute_scales, cwtd, cwtd_batched, cwtd_fftw, cwtd_mlx

# --- Test signals (always available) ---
from .signals import ucantor, tau_equation

# --- Data I/O (pandas lazy-imported inside functions) ---
# Lazy: see __getattr__ below

# --- Hurst exponent (always available) ---
from .hurst import rescaled_range, hurst_rs

# --- Numba-dependent modules: lazy re-exports ---
# These are imported on first access to avoid ImportError when numba
# is not installed or incompatible with the current NumPy version.


def __getattr__(name):
    _extrema_names = {'compute_extlis_numba', 'compute_extrep'}
    _chain_names = {
        'chain_all', 'chain_delete_all', 'chain_max_wrapper',
        'trace_chains', 'save_chain_state',
    }
    _partition_names = {
        'compute_partition_function', 'pf_get_T', 'pf_get_H', 'pf_get_D',
        'pf_standard_addition', 'pf_copy',
    }
    _spectra_names = {'compute_spectra', 'theoretical_devil_staircase'}
    _io_names = {'read_usgs_10min_file', 'create_signal', 'select_date_range'}

    if name in _extrema_names:
        from . import extrema
        return getattr(extrema, name)
    if name in _chain_names:
        from . import chains
        return getattr(chains, name)
    if name in _partition_names:
        from . import partition
        return getattr(partition, name)
    if name in _spectra_names:
        from . import spectra
        return getattr(spectra, name)
    if name in _io_names:
        from . import io
        return getattr(io, name)
    raise AttributeError(f"module 'wtmm' has no attribute {name!r}")
