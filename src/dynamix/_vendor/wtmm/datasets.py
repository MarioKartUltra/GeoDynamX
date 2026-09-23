"""Signal degradations and dataset builder for NN slope estimation training.

Provides noise injection, quantization, signal resizing, and a pipeline
that runs WTMM analysis on degraded signals to produce (features, labels)
pairs for training the slope estimator network.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Adaptive q grid
# ---------------------------------------------------------------------------

def adaptive_q_grid(q_min=-3, q_max=4, fine_min=-2, fine_max=2,
                    coarse_step=0.5, fine_step=0.1):
    """Build an adaptive q grid: fine spacing near transitions, coarse in tails.

    Parameters
    ----------
    q_min, q_max : float, outer range
    fine_min, fine_max : float, inner range with fine resolution
    coarse_step : float, spacing outside the fine region
    fine_step : float, spacing inside the fine region

    Returns
    -------
    q : sorted 1D array of unique q values
    """
    parts = []
    if q_min < fine_min:
        parts.append(np.arange(q_min, fine_min, coarse_step))
    parts.append(np.arange(fine_min, fine_max, fine_step))
    if fine_max < q_max:
        parts.append(np.arange(fine_max, q_max + coarse_step / 2, coarse_step))
    q = np.unique(np.round(np.concatenate(parts), decimals=6))
    return q


# Default q grid for phase-transition-aware analysis
DEFAULT_Q = adaptive_q_grid()  # ~47 values


# ---------------------------------------------------------------------------
# Signal degradations
# ---------------------------------------------------------------------------

def add_white_noise(signal, snr_db):
    """Add white Gaussian noise at a given SNR (dB).

    SNR = 10 * log10(P_signal / P_noise), so
    P_noise = P_signal / 10^(snr_db/10).
    """
    signal = np.asarray(signal, dtype=np.float64)
    p_signal = np.mean(signal ** 2)
    if p_signal == 0:
        return signal.copy()
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(len(signal)) * np.sqrt(p_noise)
    return signal + noise


def add_pink_noise(signal, snr_db):
    """Add 1/f (pink) noise at a given SNR (dB).

    Generated via FFT filtering: white noise with amplitude scaled by 1/sqrt(f).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    p_signal = np.mean(signal ** 2)
    if p_signal == 0:
        return signal.copy()

    # Generate pink noise via spectral shaping
    white = np.random.randn(n)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1.0  # avoid division by zero
    spectrum = np.fft.rfft(white)
    spectrum *= 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n)

    # Normalize to desired SNR
    p_pink = np.mean(pink ** 2)
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    pink *= np.sqrt(p_noise / p_pink)
    return signal + pink


def add_brownian_noise(signal, snr_db):
    """Add 1/f^2 (Brownian) noise at a given SNR (dB).

    Generated as cumulative sum of white noise, then scaled.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    p_signal = np.mean(signal ** 2)
    if p_signal == 0:
        return signal.copy()

    brown = np.cumsum(np.random.randn(n))
    brown -= np.mean(brown)

    p_brown = np.mean(brown ** 2)
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    brown *= np.sqrt(p_noise / p_brown)
    return signal + brown


def quantize(signal, n_bits):
    """Uniform quantization to n_bits resolution.

    Maps signal range to 2^n_bits levels, then reconstructs.
    """
    signal = np.asarray(signal, dtype=np.float64)
    s_min, s_max = signal.min(), signal.max()
    if s_max == s_min:
        return signal.copy()
    n_levels = 2 ** n_bits
    # Normalize to [0, 1], quantize, denormalize
    normalized = (signal - s_min) / (s_max - s_min)
    quantized = np.round(normalized * (n_levels - 1)) / (n_levels - 1)
    return quantized * (s_max - s_min) + s_min


def resize_signal(signal, target_length):
    """Resize signal to target_length by truncation or reflect-padding.

    If shorter than target: reflect-pad to fill.
    If longer: truncate from the end.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == target_length:
        return signal.copy()
    if n > target_length:
        return signal[:target_length].copy()

    # Reflect-pad
    result = np.empty(target_length, dtype=np.float64)
    result[:n] = signal
    remaining = target_length - n
    # Use numpy pad logic: reflect without edge duplication
    padded = np.pad(signal, (0, remaining), mode='reflect')
    return padded


# ---------------------------------------------------------------------------
# Degradation configuration helpers
# ---------------------------------------------------------------------------

_NOISE_FUNCS = {
    'white': add_white_noise,
    'pink': add_pink_noise,
    'brownian': add_brownian_noise,
    'none': None,
}


def apply_degradation(signal, noise_type='none', snr_db=None,
                      n_bits=None, target_length=None):
    """Apply a combination of degradations to a signal.

    Parameters
    ----------
    signal : array-like
    noise_type : str, one of 'none', 'white', 'pink', 'brownian'
    snr_db : float, optional — required if noise_type != 'none'
    n_bits : int, optional — quantize to this bit depth
    target_length : int, optional — resize signal

    Returns
    -------
    degraded : ndarray
    """
    s = np.asarray(signal, dtype=np.float64).copy()

    if target_length is not None:
        s = resize_signal(s, target_length)

    if noise_type != 'none' and snr_db is not None:
        noise_fn = _NOISE_FUNCS[noise_type]
        s = noise_fn(s, snr_db)

    if n_bits is not None:
        s = quantize(s, n_bits)

    return s


# ---------------------------------------------------------------------------
# WTMM feature extraction
# ---------------------------------------------------------------------------

def _max_n_oct(signal_length, wavelet_name, n_voice, a_min=1.0):
    """Compute maximum n_oct so the largest filter fits in the signal.

    The filter size at scale a is approximately:
        a * (|x_min_factor| + x_max_factor) * fact + 1
    The largest scale is a_min * 2^n_oct, so we need:
        a_min * 2^n_oct * support_width + 1 <= signal_length
    """
    from .wavelets import WAVELETS
    w = WAVELETS[wavelet_name]
    support_width = (-w['x_min_factor'] + w['x_max_factor']) * w['fact']
    # Max scale where filter fits
    a_max = (signal_length - 1) / support_width
    if a_max <= a_min:
        return 0
    return int(np.log2(a_max / a_min))


def _run_wtmm_pipeline(signal, wavelet='g2', n_oct=5, n_voice=10,
                        a_min=1.0, q_list=None, coi_filter=True):
    """Run the full WTMM pipeline on a signal, return partition function.

    Uses the existing wtmm package functions. Falls back through backends:
    MLX → FFTW → torch. Automatically reduces n_oct if the signal is too
    short for the requested number of octaves.

    Returns
    -------
    pf : dict or None if pipeline fails
    n_scales_valid : int, number of valid scales (index_max + 1)
    """
    if q_list is None:
        q_list = DEFAULT_Q.tolist()

    from .cwt import cwtd_mlx, cwtd_fftw, cwtd, compute_scales
    from .extrema import compute_extrep
    from .chains import chain_all, chain_delete_all, chain_max_wrapper
    from .partition import compute_partition_function

    # Adapt n_oct to signal length
    max_oct = _max_n_oct(len(signal), wavelet, n_voice, a_min)
    if max_oct < 2:
        return None, 0
    n_oct = min(n_oct, max_oct)

    n_scales = n_oct * n_voice

    # CWT — try backends in order
    try:
        coeffs, scales, valid_ranges = cwtd_mlx(
            signal, a_min=a_min, n_oct=n_oct, n_voice=n_voice,
            wavelet_name=wavelet)
    except Exception:
        try:
            coeffs, scales, valid_ranges = cwtd_fftw(
                signal, a_min=a_min, n_oct=n_oct, n_voice=n_voice,
                wavelet_name=wavelet)
        except Exception:
            coeffs, scales, valid_ranges = cwtd(
                signal, a_min=a_min, n_oct=n_oct, n_voice=n_voice,
                wavelet_name=wavelet, border='mirror')

    # COI filter
    if coi_filter and valid_ranges is not None:
        for s_idx in range(len(coeffs)):
            lo, hi = valid_ranges[s_idx]
            coeffs[s_idx][:lo] = 0.0
            coeffs[s_idx][hi:] = 0.0

    # Extrema
    try:
        extrep_abs, extrep_ord, extrep_idx = compute_extrep(
            coeffs, scales, valid_ranges=valid_ranges if coi_filter else None)
    except Exception:
        return None, 0

    # Chains
    try:
        coarser_links, finer_links = chain_all(extrep_abs, extrep_ord)
        chain_delete_all(extrep_abs, extrep_ord, extrep_idx,
                         coarser_links, finer_links)
        chain_max_wrapper(extrep_ord, coarser_links,
                          n_oct=n_oct, n_voice=n_voice, a_min=a_min)
    except Exception:
        return None, 0

    # Partition function
    try:
        pf = compute_partition_function(
            extrep_ord, n_oct, n_voice, a_min, q_list)
    except Exception:
        return None, 0

    return pf, pf['index_max'] + 1


def extract_features(pf, n_q_target=None, n_scales_target=None):
    """Extract feature tensors from a partition function dict.

    Parameters
    ----------
    pf : dict from compute_partition_function
    n_q_target : int, optional — pad/truncate q dimension
    n_scales_target : int, optional — pad/truncate scale dimension

    Returns
    -------
    features : dict with keys:
        'T_curves' : (n_q, n_scales) — T(q,a) partition function values
        'H_curves' : (n_q, n_scales) — H(q,a) values
        'D_curves' : (n_q, n_scales) — D(q,a) values
        'C_curves' : (n_q, n_scales) — C(q,a) specific heat from T curves
        'n_ext'    : (n_scales,) — number of extrema per scale
        'q_list'   : (n_q,) — q values
        'log2_a'   : (n_scales,) — log2 of scales
    """
    from .partition import pf_get_T, pf_get_H, pf_get_D

    q_array = pf['q_list']
    n_q = len(q_array)
    n_scales = pf['sTq'].shape[1]

    T_curves = np.zeros((n_q, n_scales))
    H_curves = np.zeros((n_q, n_scales))
    D_curves = np.zeros((n_q, n_scales))

    for i in range(n_q):
        T_curves[i] = pf_get_T(pf, i, mode='extensive')
        H_curves[i] = pf_get_H(pf, i, mode='extensive')
        D_curves[i] = pf_get_D(pf, i, q_array[i], mode='extensive')

    # Specific heat C(q,a) = -d²T/dq² per scale (phase transition detector)
    C_curves = compute_specific_heat(T_curves, q_array)

    # Replace NaN/inf with 0
    for arr in [T_curves, H_curves, D_curves, C_curves]:
        arr[~np.isfinite(arr)] = 0.0

    # Pad or truncate to target dimensions
    if n_q_target is not None and n_q_target != n_q:
        T_curves = _pad_or_truncate(T_curves, n_q_target, axis=0)
        H_curves = _pad_or_truncate(H_curves, n_q_target, axis=0)
        D_curves = _pad_or_truncate(D_curves, n_q_target, axis=0)
        C_curves = _pad_or_truncate(C_curves, n_q_target, axis=0)

    if n_scales_target is not None and n_scales_target != n_scales:
        T_curves = _pad_or_truncate(T_curves, n_scales_target, axis=1)
        H_curves = _pad_or_truncate(H_curves, n_scales_target, axis=1)
        D_curves = _pad_or_truncate(D_curves, n_scales_target, axis=1)
        C_curves = _pad_or_truncate(C_curves, n_scales_target, axis=1)

    n_ext = pf['n_ext'].astype(np.float64)
    if n_scales_target is not None:
        n_ext = _pad_or_truncate_1d(n_ext, n_scales_target)

    return {
        'T_curves': T_curves,
        'H_curves': H_curves,
        'D_curves': D_curves,
        'C_curves': C_curves,
        'n_ext': n_ext,
        'q_list': q_array,
        'log2_a': pf['log2_a'],
    }


def _pad_or_truncate(arr, target, axis):
    """Pad with zeros or truncate along axis."""
    current = arr.shape[axis]
    if current >= target:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target)
        return arr[tuple(slices)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(arr, pad_width, mode='constant', constant_values=0.0)


def _pad_or_truncate_1d(arr, target):
    """Pad or truncate a 1D array."""
    if len(arr) >= target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)), mode='constant', constant_values=0.0)


# ---------------------------------------------------------------------------
# q validity mask
# ---------------------------------------------------------------------------

def compute_q_valid_mask(theory, q_list):
    """Compute a boolean mask of which q values have finite analytical spectra.

    Only masks truly non-finite values. Does NOT exclude D(q) < 0 regions —
    those contain phase transition signatures (freezing, kinks) that the
    network should learn to detect.

    Parameters
    ----------
    theory : dict with 'tau_q', 'h_q', 'D_q' arrays
    q_list : array of q values

    Returns
    -------
    mask : boolean ndarray of shape (n_q,)
    """
    tau = np.asarray(theory['tau_q'], dtype=np.float64)
    h = np.asarray(theory['h_q'], dtype=np.float64)
    D = np.asarray(theory['D_q'], dtype=np.float64)

    mask = np.isfinite(tau) & np.isfinite(h) & np.isfinite(D)

    # Reject truly divergent values (numerical overflow, not freezing)
    mask &= (np.abs(h) < 1000) & (np.abs(tau) < 1000)

    return mask


def compute_specific_heat(tau_q, q_array):
    """Compute specific heat c(q) = -q² d²τ/dq² from τ(q) curve.

    From the thermodynamic analogy (Klamut et al. 2020 Eq. 14,
    also Beck & Schlögl 1993 Eq. 3):
        Temperature: q → 1/kT
        c(q) = dα/d(1/q) = -q² dα/dq = -q² d²τ/dq²

    Phase transitions show as peaks/divergences in c(q).
    c(q) > 0: thermally stable phase
    c(q) < 0: thermally unstable phase
    c(q) = 0: phase transition boundary

    Uses second-order finite differences with the q spacing from the
    (potentially adaptive) q grid.

    Parameters
    ----------
    tau_q : (n_q,) or (n_q, n_scales) array
    q_array : (n_q,) array of q values

    Returns
    -------
    C_q : same shape as tau_q, specific heat (NaN at boundaries)
    """
    tau = np.asarray(tau_q, dtype=np.float64)
    q = np.asarray(q_array, dtype=np.float64)

    if tau.ndim == 1:
        # Non-uniform second derivative via finite differences
        C = np.full_like(tau, np.nan)
        for i in range(1, len(q) - 1):
            dq_minus = q[i] - q[i - 1]
            dq_plus = q[i + 1] - q[i]
            dq_avg = (dq_minus + dq_plus) / 2.0
            d2tau = ((tau[i + 1] - tau[i]) / dq_plus
                     - (tau[i] - tau[i - 1]) / dq_minus) / dq_avg
            C[i] = -q[i] ** 2 * d2tau
        return C
    elif tau.ndim == 2:
        # (n_q, n_scales) — compute per-scale
        n_q, n_scales = tau.shape
        C = np.full_like(tau, np.nan)
        for i in range(1, n_q - 1):
            dq_minus = q[i] - q[i - 1]
            dq_plus = q[i + 1] - q[i]
            dq_avg = (dq_minus + dq_plus) / 2.0
            d2tau = ((tau[i + 1] - tau[i]) / dq_plus
                     - (tau[i] - tau[i - 1]) / dq_minus) / dq_avg
            C[i] = -q[i] ** 2 * d2tau
        return C
    else:
        raise ValueError(f'tau_q must be 1D or 2D, got {tau.ndim}D')


def detect_phase_transition(tau_q, q_array, h_q=None, D_q=None,
                            C_threshold=2.0):
    """Detect phase transition from multifractal spectra.

    Two detection methods:
    1. Specific heat peak: C(q) = -d²τ/dq² exceeds C_threshold
       (e.g. Feigenbaum attractor, sharp kinks in τ)
    2. D(h) zero-crossing: D(q) goes below 0, indicating incomplete
       spectrum / freezing transition (e.g. log-Poisson left limb)
       q* is where D first crosses zero from the right.

    Parameters
    ----------
    tau_q : (n_q,) array
    q_array : (n_q,) array
    h_q, D_q : (n_q,) arrays, optional — if provided, also check D<0
    C_threshold : float

    Returns
    -------
    has_transition : bool
    q_star : float, critical q (NaN if no transition)
    C_max : float, peak specific heat value
    """
    C_q = compute_specific_heat(tau_q, q_array)
    finite = np.isfinite(C_q)
    C_max = np.nanmax(C_q) if np.any(finite) else 0.0

    # Method 1: C(q) spike
    if C_max > C_threshold:
        idx = np.nanargmax(C_q)
        return True, q_array[idx], C_max

    # Method 2: D(q) zero-crossing (freezing transition)
    if D_q is not None:
        D = np.asarray(D_q)
        # Find rightmost q where D crosses from positive to negative
        # (scanning from high q to low q, where does D first go < 0?)
        for i in range(len(D) - 1, 0, -1):
            if D[i] >= 0 and D[i - 1] < 0:
                # Linear interpolation for q*
                frac = D[i] / (D[i] - D[i - 1])
                q_star = q_array[i] - frac * (q_array[i] - q_array[i - 1])
                return True, q_star, C_max
        # Also check if D is negative at the boundary
        if D[0] < 0 and np.any(D > 0):
            # D is negative at left edge — transition is at or before q_min
            for i in range(len(D)):
                if D[i] >= 0:
                    frac = -D[i - 1] / (D[i] - D[i - 1])
                    q_star = q_array[i - 1] + frac * (q_array[i] - q_array[i - 1])
                    return True, q_star, C_max
                    break

    return False, np.nan, C_max


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

# Default degradation grid
DEFAULT_NOISE_TYPES = ['none', 'white', 'pink', 'brownian']
DEFAULT_SNRS = [40, 30, 20, 15, 10]
DEFAULT_N_BITS = [None, 16, 14, 12, 10, 8]
DEFAULT_LENGTHS = [None]  # None = use original length
DEFAULT_WAVELETS = ['g2']
DEFAULT_N_VOICES = [10]


def build_degradation_configs(noise_types=None, snrs=None, n_bits_list=None,
                              target_lengths=None):
    """Generate all degradation parameter combinations.

    Returns
    -------
    list of dict, each with keys: noise_type, snr_db, n_bits, target_length
    """
    if noise_types is None:
        noise_types = DEFAULT_NOISE_TYPES
    if snrs is None:
        snrs = DEFAULT_SNRS
    if n_bits_list is None:
        n_bits_list = DEFAULT_N_BITS
    if target_lengths is None:
        target_lengths = DEFAULT_LENGTHS

    configs = []
    for noise_type in noise_types:
        snr_values = [None] if noise_type == 'none' else snrs
        for snr in snr_values:
            for n_bits in n_bits_list:
                for length in target_lengths:
                    configs.append({
                        'noise_type': noise_type,
                        'snr_db': snr,
                        'n_bits': n_bits,
                        'target_length': length,
                    })
    return configs


def build_pipeline_configs(wavelets=None, n_voices=None, coi_flags=None):
    """Generate all pipeline parameter combinations.

    Returns
    -------
    list of dict, each with keys: wavelet, n_voice, coi_filter
    """
    if wavelets is None:
        wavelets = DEFAULT_WAVELETS
    if n_voices is None:
        n_voices = DEFAULT_N_VOICES
    if coi_flags is None:
        coi_flags = [True]

    configs = []
    for w in wavelets:
        for nv in n_voices:
            for coi in coi_flags:
                configs.append({
                    'wavelet': w,
                    'n_voice': nv,
                    'coi_filter': coi,
                })
    return configs


def encode_metadata(deg_config, pipe_config, signal_length):
    """Encode degradation + pipeline config as a flat feature vector.

    Returns
    -------
    meta : ndarray of shape (13,)
        [wavelet_g1, wavelet_g2, wavelet_g3, wavelet_g4,
         n_voice (normalized), signal_length (log2),
         coi_flag,
         noise_none, noise_white, noise_pink, noise_brownian,
         snr_db (normalized, 0 if no noise),
         n_bits (normalized, 1.0 if no quantization)]
    """
    # Wavelet one-hot (4)
    wavelet_map = {'g1': 0, 'g2': 1, 'g3': 2, 'g4': 3}
    wav_onehot = np.zeros(4)
    wav_idx = wavelet_map.get(pipe_config['wavelet'], 1)
    wav_onehot[wav_idx] = 1.0

    # n_voice normalized (divide by 20)
    nv = pipe_config['n_voice'] / 20.0

    # Signal length (log2, normalized by 13 ~= log2(8192))
    sl = np.log2(signal_length) / 13.0

    # COI flag
    coi = 1.0 if pipe_config['coi_filter'] else 0.0

    # Noise type one-hot (4)
    noise_map = {'none': 0, 'white': 1, 'pink': 2, 'brownian': 3}
    noise_onehot = np.zeros(4)
    noise_idx = noise_map.get(deg_config['noise_type'], 0)
    noise_onehot[noise_idx] = 1.0

    # SNR normalized (0 if no noise, otherwise snr/60)
    snr = 0.0
    if deg_config['snr_db'] is not None:
        snr = deg_config['snr_db'] / 60.0

    # n_bits normalized (1.0 if None, else n_bits/16)
    bits = 1.0
    if deg_config['n_bits'] is not None:
        bits = deg_config['n_bits'] / 16.0

    return np.array([
        *wav_onehot, nv, sl, coi, *noise_onehot, snr, bits
    ], dtype=np.float64)


def encode_metadata_v2(pipe_config, signal_length, n_ext, log2_a,
                       n_bits=None):
    """Encode metadata as a 10-feature vector (real-data knowable only).

    Removes noise_type and snr_db (unknowable in real data).

    Parameters
    ----------
    pipe_config : dict with 'wavelet', 'n_voice'
    signal_length : int
    n_ext : array, number of extrema per scale
    log2_a : array, log2 of scales
    n_bits : int or None, quantization depth

    Returns
    -------
    meta : ndarray of shape (10,)
        [wavelet_g1..g4 (4), n_voice/20, log2(N)/15, n_bits/16,
         n_oct/8, log2(sum(n_ext))/20, slope of log(n_ext) vs log2(a)]
    """
    # Wavelet one-hot (4)
    wavelet_map = {'g1': 0, 'g2': 1, 'g3': 2, 'g4': 3}
    wav_onehot = np.zeros(4)
    wav_idx = wavelet_map.get(pipe_config['wavelet'], 1)
    wav_onehot[wav_idx] = 1.0

    # n_voice normalized
    nv = pipe_config['n_voice'] / 20.0

    # Signal length (log2, normalized by 15)
    sl = np.log2(max(signal_length, 1)) / 15.0

    # n_bits normalized (1.0 if unknown)
    bits = 1.0
    if n_bits is not None:
        bits = n_bits / 16.0

    # n_oct
    n_oct = (log2_a[-1] - log2_a[0]) if len(log2_a) > 1 else 0.0
    n_oct_norm = n_oct / 8.0

    # Data richness: log2(sum(n_ext)) / 20
    n_ext = np.asarray(n_ext, dtype=float)
    total_ext = max(n_ext.sum(), 1.0)
    richness = np.log2(total_ext) / 20.0

    # Extrema falloff rate: slope of log(n_ext+1) vs log2(a)
    log_nex = np.log(n_ext + 1.0)
    if len(log2_a) >= 2:
        valid = np.isfinite(log_nex) & np.isfinite(log2_a)
        if valid.sum() >= 2:
            coeffs = np.polyfit(log2_a[valid], log_nex[valid], 1)
            falloff = coeffs[0]
        else:
            falloff = 0.0
    else:
        falloff = 0.0

    return np.array([
        *wav_onehot, nv, sl, bits, n_oct_norm, richness, falloff
    ], dtype=np.float64)


def build_training_set(signals_with_theory, degradation_configs=None,
                       pipeline_configs=None, n_oct=5, a_min=1.0,
                       q_list=None, n_q_target=None, n_scales_target=None,
                       log2_a_min=None, log2_a_max=None,
                       verbose=True):
    """Build a training dataset from signals with known analytical spectra.

    Parameters
    ----------
    signals_with_theory : list of dict, each with keys:
        'name'    : str, signal name
        'signal'  : ndarray, the clean signal
        'theory'  : dict with 'tau_q', 'h_q', 'D_q' arrays (n_q,)
                    These are the analytical ground truth slopes.
        'q_list'  : array of q values matching theory arrays
        'q_valid_mask' : bool array (n_q,), optional — True where theory
                    values are physically meaningful (finite, D>=0).
                    Auto-computed via compute_q_valid_mask if not provided.
    degradation_configs : list of dict from build_degradation_configs()
    pipeline_configs : list of dict from build_pipeline_configs()
    n_oct : int, number of octaves for CWT
    a_min : float, minimum scale
    q_list : list, q values for partition function (should match theory)
    n_q_target : int, target q dimension for features (pad/truncate)
    n_scales_target : int, target scale dimension for features
    log2_a_min, log2_a_max : float, optional — if set, also compute OLS
        spectra for comparison
    verbose : bool, print progress

    Returns
    -------
    dataset : list of dict, each with:
        'features'  : dict from extract_features
        'metadata'  : ndarray (13,) encoded metadata
        'labels'    : dict with 'tau_q', 'h_q', 'D_q', 'q_valid_mask'
        'ols'       : dict with 'tau_q', 'h_q', 'D_q' (OLS spectra) or None
        'info'      : dict with signal name, degradation, pipeline params
    """
    if degradation_configs is None:
        degradation_configs = build_degradation_configs()
    if pipeline_configs is None:
        pipeline_configs = build_pipeline_configs()
    if q_list is None:
        q_list = DEFAULT_Q.tolist()

    dataset = []
    total = len(signals_with_theory) * len(degradation_configs) * len(pipeline_configs)
    count = 0
    failures = 0

    for sig_info in signals_with_theory:
        sig_name = sig_info['name']
        clean_signal = sig_info['signal']
        theory = sig_info['theory']
        sig_type = sig_info.get('type', 'function')

        # q validity mask — auto-compute if not provided
        if 'q_valid_mask' in sig_info and sig_info['q_valid_mask'] is not None:
            q_valid = np.asarray(sig_info['q_valid_mask'], dtype=bool)
        else:
            q_valid = compute_q_valid_mask(theory, q_list)

        for deg_cfg in degradation_configs:
            # Apply degradation
            degraded = apply_degradation(
                clean_signal,
                noise_type=deg_cfg['noise_type'],
                snr_db=deg_cfg['snr_db'],
                n_bits=deg_cfg['n_bits'],
                target_length=deg_cfg['target_length'],
            )

            # Integrate measure-type signals before WTMM
            if sig_type == 'measure':
                degraded = np.cumsum(degraded)

            sig_len = len(degraded)

            for pipe_cfg in pipeline_configs:
                count += 1
                if verbose and count % 50 == 0:
                    print(f'  [{count}/{total}] {sig_name} | '
                          f'{deg_cfg["noise_type"]} SNR={deg_cfg["snr_db"]} '
                          f'bits={deg_cfg["n_bits"]} | '
                          f'wav={pipe_cfg["wavelet"]} nv={pipe_cfg["n_voice"]}')

                # Run WTMM pipeline
                pf, n_valid = _run_wtmm_pipeline(
                    degraded,
                    wavelet=pipe_cfg['wavelet'],
                    n_oct=n_oct,
                    n_voice=pipe_cfg['n_voice'],
                    a_min=a_min,
                    q_list=q_list,
                    coi_filter=pipe_cfg['coi_filter'],
                )

                if pf is None or n_valid < 3:
                    failures += 1
                    continue

                # Extract features
                features = extract_features(
                    pf, n_q_target=n_q_target,
                    n_scales_target=n_scales_target)

                # Metadata vectors (v1 for backward compat, v2 for new model)
                meta = encode_metadata(deg_cfg, pipe_cfg, sig_len)
                meta_v2 = encode_metadata_v2(
                    pipe_cfg, sig_len,
                    features['n_ext'], features['log2_a'],
                    n_bits=deg_cfg.get('n_bits'))

                # OLS spectra (for comparison, not labels)
                # Canonical: h(q), D(q) from direct H(q,a), D(q,a) slopes
                # Microcanonical: tau(q) from T(q,a) slope, then Legendre
                ols = None
                if log2_a_min is not None and log2_a_max is not None:
                    try:
                        from .spectra import compute_spectra
                        ols_can = compute_spectra(
                            pf, log2_a_min, log2_a_max,
                            method='canonical')
                        ols = {
                            'tau_q': ols_can['tau_q'],
                            'h_q': ols_can['h_q'],
                            'D_q': ols_can['D_q'],
                        }
                        try:
                            ols_mic = compute_spectra(
                                pf, log2_a_min, log2_a_max,
                                method='microcanonical')
                            ols['tau_q_legendre'] = ols_mic['tau_q']
                            ols['h_q_legendre'] = ols_mic['h_q']
                            ols['D_q_legendre'] = ols_mic['D_q']
                        except Exception:
                            pass
                    except Exception:
                        pass

                # Phase transition detection from analytical spectra
                tau_theory = np.array(theory['tau_q'])
                h_theory = np.array(theory['h_q'])
                D_theory = np.array(theory['D_q'])
                q_arr = np.array(q_list)
                has_trans, q_star, C_max = detect_phase_transition(
                    tau_theory, q_arr, h_q=h_theory, D_q=D_theory)
                C_theory = compute_specific_heat(tau_theory, q_arr)
                C_theory[~np.isfinite(C_theory)] = 0.0

                dataset.append({
                    'features': features,
                    'metadata': meta,
                    'metadata_v2': meta_v2,
                    'labels': {
                        'tau_q': tau_theory,
                        'h_q': np.array(theory['h_q']),
                        'D_q': np.array(theory['D_q']),
                        'C_q': C_theory,
                        'q_valid_mask': q_valid.copy(),
                        'has_transition': has_trans,
                        'q_star': q_star,
                    },
                    'ols': ols,
                    'info': {
                        'signal_name': sig_name,
                        'degradation': deg_cfg,
                        'pipeline': pipe_cfg,
                        'signal_length': sig_len,
                    },
                })

    if verbose:
        print(f'Dataset built: {len(dataset)} samples '
              f'({failures} pipeline failures out of {total} attempts)')

    return dataset


def dataset_to_arrays(dataset, n_q, n_scales):
    """Convert dataset list to numpy arrays suitable for training.

    Parameters
    ----------
    dataset : list of dict from build_training_set
    n_q : int, expected q dimension
    n_scales : int, expected scale dimension

    Returns
    -------
    X_curves : (N, 4, n_q, n_scales) — T, H, D, C curves stacked
    X_meta   : (N, 13) — metadata vectors
    Y        : (N, 3, n_q) — tau, h, D labels stacked (0 where invalid)
    Y_C      : (N, n_q) — analytical specific heat C(q) labels
    Y_trans  : (N, 2) — [has_transition (0/1), q_star (NaN if none)]
    Q_mask   : (N, n_q) — float, 1.0 where labels are valid
    Y_ols    : (N, 3, n_q) or None — OLS comparison if available
    """
    N = len(dataset)
    X_curves = np.zeros((N, 4, n_q, n_scales), dtype=np.float32)
    X_meta = np.zeros((N, 13), dtype=np.float32)
    Y = np.zeros((N, 3, n_q), dtype=np.float32)
    Y_C = np.zeros((N, n_q), dtype=np.float32)
    Y_trans = np.zeros((N, 2), dtype=np.float32)
    Q_mask = np.ones((N, n_q), dtype=np.float32)  # default: all valid

    has_ols = dataset[0]['ols'] is not None
    Y_ols = np.zeros((N, 3, n_q), dtype=np.float32) if has_ols else None

    for i, sample in enumerate(dataset):
        feat = sample['features']
        # Ensure correct shape — 4 channels: T, H, D, C
        for ch, key in enumerate(['T_curves', 'H_curves', 'D_curves', 'C_curves']):
            arr = _pad_or_truncate(_pad_or_truncate(
                feat[key], n_q, axis=0), n_scales, axis=1)
            X_curves[i, ch] = arr

        X_meta[i] = sample['metadata']

        labels = sample['labels']
        for j, key in enumerate(['tau_q', 'h_q', 'D_q']):
            arr = np.array(labels[key], dtype=np.float32)
            if len(arr) >= n_q:
                Y[i, j] = arr[:n_q]
            else:
                Y[i, j, :len(arr)] = arr

        # Analytical specific heat
        if 'C_q' in labels:
            c_arr = np.array(labels['C_q'], dtype=np.float32)
            if len(c_arr) >= n_q:
                Y_C[i] = c_arr[:n_q]
            else:
                Y_C[i, :len(c_arr)] = c_arr

        # Phase transition labels
        Y_trans[i, 0] = 1.0 if labels.get('has_transition', False) else 0.0
        q_star = labels.get('q_star', np.nan)
        Y_trans[i, 1] = q_star if np.isfinite(q_star) else 0.0

        # q validity mask
        if 'q_valid_mask' in labels:
            mask = np.asarray(labels['q_valid_mask'], dtype=np.float32)
            if len(mask) >= n_q:
                Q_mask[i] = mask[:n_q]
            else:
                Q_mask[i, :len(mask)] = mask
                Q_mask[i, len(mask):] = 0.0

        # Zero out labels at invalid q to avoid NaN/inf in training
        for j in range(3):
            Y[i, j] *= Q_mask[i]
        Y_C[i] *= Q_mask[i]

        if has_ols and sample['ols'] is not None:
            for j, key in enumerate(['tau_q', 'h_q', 'D_q']):
                arr = np.array(sample['ols'][key], dtype=np.float32)
                if len(arr) >= n_q:
                    Y_ols[i, j] = arr[:n_q]
                else:
                    Y_ols[i, j, :len(arr)] = arr

    return X_curves, X_meta, Y, Y_C, Y_trans, Q_mask, Y_ols


def dataset_to_arrays_v2(dataset, n_q, n_scales):
    """Convert dataset to arrays for the v2 ScaleWeightEstimator.

    Like dataset_to_arrays but uses v2 metadata (10 features), adds n_ext
    as a 5th input channel, and returns log2_a per sample.

    Returns
    -------
    X_curves : (N, 5, n_q, n_scales) — T, H, D, C, n_ext channels
    X_meta   : (N, 10) — v2 metadata vectors
    Y        : (N, 3, n_q) — tau, h, D labels
    Y_C      : (N, n_q) — analytical C(q)
    Y_trans  : (N, 2) — [has_transition, q_star]
    Q_mask   : (N, n_q) — validity mask
    X_log2_a : (N, n_scales) — log2 of scales per sample
    Y_ols    : (N, 3, n_q) or None
    """
    N = len(dataset)
    X_curves = np.zeros((N, 5, n_q, n_scales), dtype=np.float32)
    X_meta = np.zeros((N, 10), dtype=np.float32)
    Y = np.zeros((N, 3, n_q), dtype=np.float32)
    Y_C = np.zeros((N, n_q), dtype=np.float32)
    Y_trans = np.zeros((N, 2), dtype=np.float32)
    Q_mask = np.ones((N, n_q), dtype=np.float32)
    X_log2_a = np.zeros((N, n_scales), dtype=np.float32)

    has_ols = dataset[0]['ols'] is not None
    Y_ols = np.zeros((N, 3, n_q), dtype=np.float32) if has_ols else None

    for i, sample in enumerate(dataset):
        feat = sample['features']

        # 4 curve channels: T, H, D, C
        for ch, key in enumerate(['T_curves', 'H_curves', 'D_curves', 'C_curves']):
            arr = _pad_or_truncate(_pad_or_truncate(
                feat[key], n_q, axis=0), n_scales, axis=1)
            X_curves[i, ch] = arr

        # 5th channel: n_ext broadcast across q
        n_ext = _pad_or_truncate_1d(feat['n_ext'], n_scales).astype(np.float32)
        # Normalize: log2(n_ext + 1) / 20
        n_ext_norm = np.log2(n_ext + 1.0) / 20.0
        X_curves[i, 4] = np.broadcast_to(n_ext_norm, (n_q, n_scales))

        # log2_a
        log2_a = _pad_or_truncate_1d(feat['log2_a'], n_scales).astype(np.float32)
        X_log2_a[i] = log2_a

        # v2 metadata
        if 'metadata_v2' in sample:
            X_meta[i] = sample['metadata_v2']
        else:
            # Fallback: compute from features
            pipe_cfg = sample['info']['pipeline']
            sig_len = sample['info']['signal_length']
            n_bits = sample['info']['degradation'].get('n_bits')
            X_meta[i] = encode_metadata_v2(
                pipe_cfg, sig_len, feat['n_ext'], feat['log2_a'],
                n_bits=n_bits)

        # Labels (same as v1)
        labels = sample['labels']
        for j, key in enumerate(['tau_q', 'h_q', 'D_q']):
            arr = np.array(labels[key], dtype=np.float32)
            if len(arr) >= n_q:
                Y[i, j] = arr[:n_q]
            else:
                Y[i, j, :len(arr)] = arr

        if 'C_q' in labels:
            c_arr = np.array(labels['C_q'], dtype=np.float32)
            if len(c_arr) >= n_q:
                Y_C[i] = c_arr[:n_q]
            else:
                Y_C[i, :len(c_arr)] = c_arr

        Y_trans[i, 0] = 1.0 if labels.get('has_transition', False) else 0.0
        q_star = labels.get('q_star', np.nan)
        Y_trans[i, 1] = q_star if np.isfinite(q_star) else 0.0

        if 'q_valid_mask' in labels:
            mask = np.asarray(labels['q_valid_mask'], dtype=np.float32)
            if len(mask) >= n_q:
                Q_mask[i] = mask[:n_q]
            else:
                Q_mask[i, :len(mask)] = mask
                Q_mask[i, len(mask):] = 0.0

        for j in range(3):
            Y[i, j] *= Q_mask[i]
        Y_C[i] *= Q_mask[i]

        if has_ols and sample['ols'] is not None:
            for j, key in enumerate(['tau_q', 'h_q', 'D_q']):
                arr = np.array(sample['ols'][key], dtype=np.float32)
                if len(arr) >= n_q:
                    Y_ols[i, j] = arr[:n_q]
                else:
                    Y_ols[i, j, :len(arr)] = arr

    return X_curves, X_meta, Y, Y_C, Y_trans, Q_mask, X_log2_a, Y_ols
