"""WTMM (Wavelet Transform Modulus Maxima) 1D analysis package.

Extracted from shared notebook code for reuse across multiple backends
(torch GPU, FFTW CPU, MLX Apple Silicon, compressed sensing).

Modules with optional dependencies (numba, torch, pyfftw, mlx) use lazy
imports — they are only loaded when you import from them explicitly.
"""

# GeoDynamix_Beta trim (spec 09, 2026-10-15): the package carries only the modules the app
# loads -- wavelets, cwt, signals, extrema, chains, partition, dupont_venugopal. The source
# package as copied from Creep/wavelet d73ba61 is at tag vendored-snapshot-2026-10.

# --- Wavelets (always available) ---
from .wavelets import WAVELETS, wavelet_direct, wavelet_support

# --- CWT engines (torch/pyfftw/mlx lazy-imported inside functions) ---
from .cwt import compute_scales, cwtd, cwtd_batched, cwtd_fftw, cwtd_mlx

# --- Test signals (always available) ---
from .signals import ucantor, tau_equation

# --- Numba-dependent modules: lazy re-exports ---
# These are imported on first access to avoid ImportError when numba
# is not installed or incompatible with the current NumPy version.


def __getattr__(name):
    _extrema_names = {'compute_extlis_numba', 'compute_extrep', 'compute_zerorep'}
    _chain_names = {
        'chain_all', 'chain_delete_all', 'chain_max_wrapper',
        'trace_chains', 'save_chain_state',
    }
    _partition_names = {
        'compute_partition_function', 'pf_get_T', 'pf_get_H', 'pf_get_D',
        'pf_get_var_T', 'pf_get_var_H', 'pf_get_var_D',
        'pf_standard_addition', 'pf_copy',
    }

    if name in _extrema_names:
        from . import extrema
        return getattr(extrema, name)
    if name in _chain_names:
        from . import chains
        return getattr(chains, name)
    if name in _partition_names:
        from . import partition
        return getattr(partition, name)
    raise AttributeError(f"module 'wtmm' has no attribute {name!r}")
