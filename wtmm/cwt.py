"""Continuous Wavelet Transform engines with multiple backends."""

import os
import numpy as np

from .wavelets import WAVELETS, wavelet_direct


def _build_filter_vectorized(sizeTot, sizeG, a, wavelet_name):
    """Build reversed wavelet filter using vectorized numpy (no Python loop)."""
    w = WAVELETS[wavelet_name]
    i_arr = np.arange(sizeTot, dtype=np.float64)
    x_arr = i_arr - sizeG
    u_arr = x_arr / (a * w['fact'])
    filt = w['func'](u_arr)[::-1].copy()
    return filt


def compute_scales(a_min, n_oct, n_voice):
    """Generate scale array: a_min * 2^(i/n_voice) for i in 0..n_oct*n_voice-1.

    Parameters
    ----------
    a_min : float
        Minimum scale.
    n_oct : int
        Number of octaves.
    n_voice : int
        Number of voices per octave.

    Returns
    -------
    scales : 1D numpy array of shape (n_oct * n_voice,)
    """
    factor = 2.0 ** (1.0 / n_voice)
    n_scales = n_oct * n_voice
    return np.array([a_min * factor**i for i in range(n_scales)])


# ---------------------------------------------------------------------------
# Torch backend (direct convolution)
# ---------------------------------------------------------------------------

def cwtd(signal, a_min, n_oct, n_voice, wavelet_name='g2', expo=-1.0, border='mirror'):
    """Continuous wavelet transform matching LastWave's CWtd() exactly.

    Uses torch conv1d with reflect padding. All computation in float64.

    Parameters
    ----------
    signal : 1D numpy array
    a_min : float
    n_oct, n_voice : int
    wavelet_name : str
    expo : float, normalization exponent (multiply by a^expo after convolution)
    border : str, padding mode ('mirror')

    Returns
    -------
    coeffs : 2D numpy array (float64), shape (n_scales, signal_size)
    scales : 1D numpy array of scale values
    valid_ranges : list of (firstp, lastp) tuples per scale
    """
    import torch

    size = len(signal)
    factor = 2.0 ** (1.0 / n_voice)
    n_scales = n_oct * n_voice

    w = WAVELETS[wavelet_name]
    d_x_min = w['x_min_factor'] * w['fact']  # negative
    d_x_max = w['x_max_factor'] * w['fact']  # positive

    # Detect device
    if torch.backends.mps.is_available():
        cwt_device = torch.device('mps')
    elif torch.cuda.is_available():
        cwt_device = torch.device('cuda')
    else:
        cwt_device = torch.device('cpu')

    # Build all filters
    scales = np.zeros(n_scales)
    filters = []
    filter_sizes = []

    a = float(a_min)
    for idx in range(n_scales):
        scales[idx] = a

        sizeD = int(a * d_x_max)
        sizeG = int(-a * d_x_min)
        sizeTot = sizeD + sizeG + 1

        if sizeTot > size:
            raise ValueError(f'Filter size {sizeTot} > signal size {size} at scale {a:.2f}')

        filt = _build_filter_vectorized(sizeTot, sizeG, a, wavelet_name)

        filters.append(filt)
        filter_sizes.append((sizeTot, sizeD, sizeG))
        a *= factor

    coeffs = np.zeros((n_scales, size))
    valid_ranges = []
    sig_np = signal.copy()

    for idx in range(n_scales):
        sizeTot, sizeD, sizeG = filter_sizes[idx]
        filt = filters[idx]

        # Valid range (matching LastWave's cv_compute with mirror padding)
        firstp = sizeG
        lastp = size - 1 - sizeD
        valid_ranges.append((firstp, lastp))

        # Float64 on cwt_device (MPS if supported, else CPU)
        sig_t = torch.tensor(sig_np, dtype=torch.float64, device=cwt_device).unsqueeze(0).unsqueeze(0)
        sig_padded = torch.nn.functional.pad(sig_t, (sizeG, sizeD), mode='reflect')

        filt_t = torch.tensor(filt, dtype=torch.float64, device=cwt_device).unsqueeze(0).unsqueeze(0)

        result = torch.nn.functional.conv1d(sig_padded, filt_t)

        mult = scales[idx] ** expo
        coeffs[idx] = (result.squeeze() * mult).cpu().numpy()

    return coeffs, scales, valid_ranges


def cwtd_batched(signal, a_min, n_oct, n_voice, wavelet_name='g2', expo=-1.0):
    """Optimized batched CWT -- groups scales by filter size for parallelism.
    All computation in float64 on best available torch device.

    Returns
    -------
    coeffs : 2D numpy array (float64), shape (n_scales, signal_size)
    scales : 1D numpy array of scale values
    valid_ranges : list of (firstp, lastp) tuples per scale
    """
    import torch
    from collections import defaultdict

    size = len(signal)
    factor = 2.0 ** (1.0 / n_voice)
    n_scales = n_oct * n_voice

    w = WAVELETS[wavelet_name]
    d_x_min = w['x_min_factor'] * w['fact']
    d_x_max = w['x_max_factor'] * w['fact']

    # Detect device
    if torch.backends.mps.is_available():
        cwt_device = torch.device('mps')
    elif torch.cuda.is_available():
        cwt_device = torch.device('cuda')
    else:
        cwt_device = torch.device('cpu')

    scales = np.array([a_min * factor**i for i in range(n_scales)])

    # Build filters and valid ranges
    all_filters = []
    all_sizeG = []
    all_sizeD = []
    valid_ranges = []
    for idx, a in enumerate(scales):
        sizeD = int(a * d_x_max)
        sizeG = int(-a * d_x_min)
        sizeTot = sizeD + sizeG + 1
        if sizeTot > size:
            raise ValueError(f'Filter size {sizeTot} > signal size {size} at scale {a:.2f}')
        filt = _build_filter_vectorized(sizeTot, sizeG, a, wavelet_name)
        all_filters.append(filt)
        all_sizeG.append(sizeG)
        all_sizeD.append(sizeD)
        valid_ranges.append((sizeG, size - 1 - sizeD))

    # Group scales by (sizeG, sizeD) for batched processing
    groups = defaultdict(list)
    for idx in range(n_scales):
        key = (all_sizeG[idx], all_sizeD[idx])
        groups[key].append(idx)

    coeffs = np.zeros((n_scales, size))
    sig_np = signal.copy()

    for (sizeG, sizeD), indices in groups.items():
        # Pad signal once for this group -- float64 on cwt_device
        sig_t = torch.tensor(sig_np, dtype=torch.float64, device=cwt_device).unsqueeze(0).unsqueeze(0)
        sig_padded = torch.nn.functional.pad(sig_t, (sizeG, sizeD), mode='reflect')
        batch_size = len(indices)
        sig_batch = sig_padded.expand(batch_size, 1, -1)

        # Stack filters, pad to same length within group
        max_len = max(all_filters[i].shape[0] for i in indices)
        weight = torch.zeros(batch_size, 1, max_len, dtype=torch.float64, device=cwt_device)
        for j, idx in enumerate(indices):
            filt = all_filters[idx]
            weight[j, 0, :len(filt)] = torch.tensor(filt, dtype=torch.float64)

        # Grouped conv1d
        sig_grouped = sig_batch.reshape(1, batch_size, -1)
        result = torch.nn.functional.conv1d(sig_grouped, weight, groups=batch_size)
        result = result.squeeze(0)  # (batch_size, size)

        # Convert to numpy float64
        result_cpu = result.cpu().numpy()
        for j, idx in enumerate(indices):
            mult = scales[idx] ** expo
            coeffs[idx] = result_cpu[j] * mult

    return coeffs, scales, valid_ranges


# ---------------------------------------------------------------------------
# FFTW overlap-save backend
# ---------------------------------------------------------------------------

def _next_power_of_2(n):
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _get_part_mirror(signal_data, signal_size, part_begin, part_size):
    """Extract a part of the signal with mirror border handling.

    Exact translation of LastWave's _get_part_r_mi_() from cv_misc.c.
    Left border:  signal[-j] for j < 0  (reflect, no edge repeat)
    Right border: reversed copy then normal copy, alternating
    """
    part = np.empty(part_size, dtype=np.float64)
    i = 0
    j = part_begin

    # Left border
    while j < 0:
        part[i] = signal_data[-j]
        i += 1
        j += 1

    # Middle
    tmp_n = min(part_size - i, signal_size - j)
    if tmp_n > 0:
        part[i:i + tmp_n] = signal_data[j:j + tmp_n]
        i += tmp_n
        j += tmp_n

    # Right border: alternating mirrored and normal copies
    while i < part_size:
        # Mirrored (reversed) copy
        tmp_n = min(part_size - i, signal_size)
        k = signal_size - 1
        for _ in range(tmp_n):
            if i >= part_size:
                break
            part[i] = signal_data[k]
            k -= 1
            i += 1

        # Normal copy
        if i < part_size:
            tmp_n = min(part_size - i, signal_size)
            part[i:i + tmp_n] = signal_data[:tmp_n]
            i += tmp_n

    return part


def _periodic_extend_filter(filter_data, filter_begin, filter_end, new_size):
    """Periodic-extend filter, matching cv_pure_periodic_extend_() from cv_misc.c.

    filter_data: array of length (filter_end - filter_begin + 1)
    filter_begin: negative (= -sizeD)
    filter_end: positive (= sizeG)

    source = filter_data offset so source[0..end] and source[begin..-1] are valid.
    Result[0..end]                    = source[0..end]         (positive part)
    Result[end+1..new_size+begin-1]   = 0                      (zero fill)
    Result[new_size+begin..new_size-1] = source[begin..-1]     (negative part wrapped)
    """
    result = np.zeros(new_size, dtype=np.float64)
    # source[i] = filter_data[i - filter_begin]
    offset = -filter_begin  # = sizeD

    # Positive part: result[0..filter_end]
    result[:filter_end + 1] = filter_data[offset:offset + filter_end + 1]

    # Negative part: result[new_size+filter_begin..new_size-1]
    neg_start = new_size + filter_begin
    neg_len = -filter_begin  # = sizeD
    result[neg_start:new_size] = filter_data[:neg_len]

    return result


def cwtd_fftw(signal, a_min, n_oct, n_voice, wavelet_name='g2', expo=-1.0):
    """CWT using FFTW overlap-save, matching LastWave's cv_n_real_mp() exactly.

    Returns
    -------
    coeffs : 2D numpy array (float64), shape (n_scales, signal_size)
    scales : 1D numpy array of scale values
    valid_ranges : list of (firstp, lastp) tuples per scale
    """
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(300)

    FFTW_THREADS = max(1, os.cpu_count() // 2)

    _rfft  = lambda x: pyfftw.interfaces.numpy_fft.rfft(x, threads=FFTW_THREADS)
    _irfft = lambda x, n: pyfftw.interfaces.numpy_fft.irfft(x, n=n, threads=FFTW_THREADS)

    size = len(signal)
    factor = 2.0 ** (1.0 / n_voice)
    n_scales = n_oct * n_voice

    w = WAVELETS[wavelet_name]
    d_x_min = w['x_min_factor'] * w['fact']  # negative
    d_x_max = w['x_max_factor'] * w['fact']  # positive

    coeffs = np.zeros((n_scales, size), dtype=np.float64)
    scales_arr = np.zeros(n_scales, dtype=np.float64)
    valid_ranges = []

    a = float(a_min)
    for idx in range(n_scales):
        scales_arr[idx] = a

        sizeD = int(a * d_x_max)
        sizeG = int(-a * d_x_min)
        sizeTot = sizeD + sizeG + 1

        if sizeTot > size:
            raise ValueError(f'Filter size {sizeTot} > signal size {size} at scale {a:.2f}')

        # Valid range (matching cwt1d.c firstp/lastp calculation)
        firstp = sizeG
        lastp = size - 1 - sizeD
        valid_ranges.append((firstp, lastp))

        # Build reversed filter (vectorized, matches CWtd exactly)
        filt = _build_filter_vectorized(sizeTot, sizeG, a, wavelet_name)

        # Filter index range: begin = -sizeD (negative), end = sizeG (positive)
        filter_begin = -sizeD
        filter_end = sizeG

        # Overlap-save parameters (matching cv_n_real_mp)
        part_size = _next_power_of_2(2 * sizeTot)
        size_of_exact_data = part_size - sizeTot + 1

        # Periodic-extend filter and FFT once
        filt_ext = _periodic_extend_filter(filt, filter_begin, filter_end, part_size)
        filt_ft = _rfft(filt_ext)

        # Process signal in overlapping parts
        nb_of_parts = int(np.ceil(size / size_of_exact_data))
        result = np.zeros(size, dtype=np.float64)

        for part_nb in range(nb_of_parts):
            part_begin = part_nb * size_of_exact_data - filter_end

            # Extract part with mirror border
            sig_part = _get_part_mirror(signal, size, part_begin, part_size)

            # FFT, multiply, IFFT (normalization handled by irfft)
            sig_ft = _rfft(sig_part)
            prod_ft = sig_ft * filt_ft
            conv_result = _irfft(prod_ft, part_size)

            # Copy exact (non-contaminated) portion to result
            copy_start = part_nb * size_of_exact_data
            if part_nb < nb_of_parts - 1:
                copy_len = size_of_exact_data
            else:
                copy_len = size - copy_start

            result[copy_start:copy_start + copy_len] = \
                conv_result[filter_end:filter_end + copy_len]

        # Apply a^expo normalization (matching CWtd)
        mult = a ** expo
        coeffs[idx] = result * mult

        a *= factor

    return coeffs, scales_arr, valid_ranges


# ---------------------------------------------------------------------------
# MLX overlap-save backend
# ---------------------------------------------------------------------------

def cwtd_mlx(signal, a_min, n_oct, n_voice, wavelet_name='g2', expo=-1.0):
    """CWT using MLX overlap-save, same algorithm as cwtd_fftw but with mlx.core FFT.

    Returns numpy arrays for interoperability.

    Returns
    -------
    coeffs : 2D numpy array (float64), shape (n_scales, signal_size)
    scales : 1D numpy array of scale values
    valid_ranges : list of (firstp, lastp) tuples per scale
    """
    import mlx.core as mx

    size = len(signal)
    factor = 2.0 ** (1.0 / n_voice)
    n_scales = n_oct * n_voice

    w = WAVELETS[wavelet_name]
    d_x_min = w['x_min_factor'] * w['fact']  # negative
    d_x_max = w['x_max_factor'] * w['fact']  # positive

    coeffs = np.zeros((n_scales, size), dtype=np.float64)
    scales_arr = np.zeros(n_scales, dtype=np.float64)
    valid_ranges = []

    a = float(a_min)
    for idx in range(n_scales):
        scales_arr[idx] = a

        sizeD = int(a * d_x_max)
        sizeG = int(-a * d_x_min)
        sizeTot = sizeD + sizeG + 1

        if sizeTot > size:
            raise ValueError(f'Filter size {sizeTot} > signal size {size} at scale {a:.2f}')

        # Valid range (matching cwt1d.c firstp/lastp calculation)
        firstp = sizeG
        lastp = size - 1 - sizeD
        valid_ranges.append((firstp, lastp))

        # Build reversed filter (vectorized, matches CWtd exactly)
        filt = _build_filter_vectorized(sizeTot, sizeG, a, wavelet_name)

        # Filter index range: begin = -sizeD (negative), end = sizeG (positive)
        filter_begin = -sizeD
        filter_end = sizeG

        # Overlap-save parameters (matching cv_n_real_mp)
        part_size = _next_power_of_2(2 * sizeTot)
        size_of_exact_data = part_size - sizeTot + 1

        # Periodic-extend filter and FFT once (via MLX)
        filt_ext = _periodic_extend_filter(filt, filter_begin, filter_end, part_size)
        filt_mx = mx.array(filt_ext)
        filt_ft = mx.fft.rfft(filt_mx)

        # Process signal in overlapping parts
        nb_of_parts = int(np.ceil(size / size_of_exact_data))
        result = np.zeros(size, dtype=np.float64)

        for part_nb in range(nb_of_parts):
            part_begin = part_nb * size_of_exact_data - filter_end

            # Extract part with mirror border (numpy, then convert)
            sig_part = _get_part_mirror(signal, size, part_begin, part_size)
            sig_mx = mx.array(sig_part)

            # FFT, multiply, IFFT
            sig_ft = mx.fft.rfft(sig_mx)
            prod_ft = sig_ft * filt_ft
            conv_result = mx.fft.irfft(prod_ft, n=part_size)

            # Evaluate and copy exact (non-contaminated) portion to result
            conv_np = np.array(conv_result)
            copy_start = part_nb * size_of_exact_data
            if part_nb < nb_of_parts - 1:
                copy_len = size_of_exact_data
            else:
                copy_len = size - copy_start

            result[copy_start:copy_start + copy_len] = \
                conv_np[filter_end:filter_end + copy_len]

        # Apply a^expo normalization (matching CWtd)
        mult = a ** expo
        coeffs[idx] = result * mult

        a *= factor

    return coeffs, scales_arr, valid_ranges
