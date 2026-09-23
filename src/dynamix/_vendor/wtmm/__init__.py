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
    _spectra_names = {'compute_spectra', 'plot_spectra', 'plot_partition_functions',
                       'print_spectra_comparison', 'theoretical_devil_staircase'}
    _io_names = {'read_usgs_10min_file', 'create_signal', 'select_date_range'}
    _dataset_names = {
        'add_white_noise', 'add_pink_noise', 'add_brownian_noise',
        'quantize', 'resize_signal', 'apply_degradation',
        'build_degradation_configs', 'build_pipeline_configs',
        'build_training_set', 'dataset_to_arrays', 'dataset_to_arrays_v2',
        'extract_features',
        'encode_metadata', 'encode_metadata_v2', 'compute_q_valid_mask',
        'adaptive_q_grid', 'DEFAULT_Q',
        'compute_specific_heat', 'detect_phase_transition',
    }
    _catalog_names = {
        'build_signal_catalog',
        'fbm_spectral', 'weierstrass_fn', 'multinomial_measure',
        'random_cascade_batch', 'feigenbaum_orbit',
        'lognormal_batch', 'logpoisson_batch', 'loggamma_batch',
        'compound_poisson_batch', 'uniform_batch', 'stable_batch',
        'cantor_tau', 'cascade_tau', 'legendre',
    }
    _local_holder_names = {
        'compute_structure_functions', 'fit_zeta_in_range',
        'compute_active_sets', 'calibrate_c_p',
        'compute_local_holder', 'holder_histogram',
        'plot_structure_functions', 'plot_zeta',
        'plot_local_holder', 'plot_holder_vs_spectrum',
    }
    _perline_names = {
        'compute_perline_partition', 'classify_lines',
        'compute_subset_spectra', 'detect_phase_transition_perline',
        'plot_line_classification', 'plot_weight_evolution',
        'perline_to_pf', 'plot_weight_skeleton', 'plot_weight_scalogram',
    }
    _nn_model_names = {
        'SlopeEstimator', 'compute_loss_with_q',
        'ScaleWeightEstimator', 'ScaleWeightHead',
        'DifferentiableWeightedOLS', 'compute_combined_loss',
    }
    _interactive_names = {'interactive_spectra', 'interactive_entropy'}
    _entropy_names = {
        'renyi_curves', 'tsallis_curves', 'shannon_entropy_per_scale',
        'compute_entropy_spectra',
        'hanel_thurner_exponents', 'lesche_distance',
        'plot_entropy_curves', 'plot_entropy_spectra',
    }
    _nn_train_names = {
        'train', 'evaluate', 'compare_nn_vs_ols', 'compute_metrics',
        'spectrum_width', 'per_signal_metrics', 'save_model', 'load_model',
        'train_v2', 'evaluate_v2',
        'plot_scale_weights', 'plot_weight_summary',
        'save_model_v2', 'load_model_v2',
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
    if name in _spectra_names:
        from . import spectra
        return getattr(spectra, name)
    if name in _io_names:
        from . import io
        return getattr(io, name)
    if name in _dataset_names:
        from . import datasets
        return getattr(datasets, name)
    if name in _catalog_names:
        from . import catalog
        return getattr(catalog, name)
    if name in _interactive_names:
        if name == 'interactive_entropy':
            from . import entropy_measures
            return getattr(entropy_measures, name)
        from . import interactive
        return getattr(interactive, name)
    if name in _entropy_names:
        from . import entropy_measures
        return getattr(entropy_measures, name)
    if name in _local_holder_names:
        from . import local_holder
        return getattr(local_holder, name)
    if name in _perline_names:
        from . import perline
        return getattr(perline, name)
    if name in _nn_model_names:
        from . import nn_model
        return getattr(nn_model, name)
    if name in _nn_train_names:
        from . import nn_train
        return getattr(nn_train, name)
    raise AttributeError(f"module 'wtmm' has no attribute {name!r}")
