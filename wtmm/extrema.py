"""WTMM extrema detection (modulus maxima per scale)."""

import numpy as np


def _compile_extrema_functions():
    """Lazy-compile numba-jitted extrema functions on first use."""
    from numba import njit

    @njit(cache=True)
    def _plateau(Y, size, ideb, eps):
        """Find end of plateau. Returns (ifin, min_val, max_val)."""
        min_val = Y[ideb]
        max_val = Y[ideb]
        for i in range(ideb, size):
            if Y[i] > max_val:
                if (Y[i] - min_val) > eps:
                    return i, min_val, max_val
                else:
                    max_val = Y[i]
            elif Y[i] < min_val:
                if (max_val - Y[i]) > eps:
                    return i, min_val, max_val
                else:
                    min_val = Y[i]
        return size, min_val, max_val

    @njit(cache=True)
    def _sign(x):
        if x > 0: return 1
        elif x < 0: return -1
        return 0

    @njit(cache=True)
    def _plateau_is_maxima(prev_val, value, next_val, min_val, max_val):
        """Test if plateau is a modulus maximum."""
        derI = _sign(value - prev_val)
        derNextI = _sign(next_val - value)
        if min_val * max_val <= 0:
            return False
        if derI != derNextI:
            if ((derI == 1 or derNextI == -1) and value > 0) or \
               ((derI == -1 or derNextI == 1) and value < 0):
                if derI != 0 and derNextI != 0:
                    return True
        return False

    @njit(cache=True)
    def _compute_extlis_numba(Y, dx, x0, epsilon, firstp, lastp):
        """Compute extrema of wavelet coefficients at one scale.
        Returns (abscissa_arr, ordinate_arr, index_arr) as flat numpy arrays.
        Matches LastWave's ComputeExtlis() with flagCausal=NO, flagInterpol=YES.

        firstp, lastp: valid range for L2 norm computation (from CWT border handling).
        """
        size = len(Y)

        # Compute threshold = epsilon * L2_norm over valid range [firstp, lastp]
        NL2 = 0.0
        for k in range(firstp, lastp + 1):
            NL2 += Y[k] * Y[k]
        NL2 = np.sqrt(NL2)
        threshold = NL2 * epsilon

        # Pre-allocate output (max possible = size/2)
        max_ext = size // 2 + 1
        abs_out = np.empty(max_ext)
        ord_out = np.empty(max_ext)
        idx_out = np.empty(max_ext, dtype=np.int64)
        count = 0

        firstI = 1
        lastI = size - 1

        i = firstI
        while i < lastI - 2:
            ifin, min_val, max_val = _plateau(Y, size, i, threshold)
            iext = (i + ifin - 1) // 2

            prevValueI = Y[i - 1]
            valueI = max_val if max_val >= 0 else min_val

            if ifin >= size:
                i = ifin
                continue

            nextValueI = Y[ifin]

            if _plateau_is_maxima(prevValueI, valueI, nextValueI, min_val, max_val):
                # Parabolic interpolation
                a_coeff = (nextValueI + prevValueI) / 2.0 - valueI

                if a_coeff != 0:
                    b_coeff = (nextValueI - prevValueI) / 2.0
                    c_coeff = valueI
                    abscissa = -b_coeff / (2 * a_coeff) + i
                    ordinate = c_coeff - b_coeff * b_coeff / (4 * a_coeff)

                    if abscissa <= i - 1 or abscissa >= ifin:
                        abscissa = float(iext)
                        ordinate = valueI
                else:
                    abscissa = float(iext)
                    ordinate = valueI

                abs_out[count] = abscissa * dx + x0
                ord_out[count] = ordinate
                idx_out[count] = iext
                count += 1

            i = ifin

        return abs_out[:count].copy(), ord_out[:count].copy(), idx_out[:count].copy()

    return _compute_extlis_numba


_extlis_jit = None


def compute_extlis_numba(Y, dx, x0, epsilon, firstp, lastp):
    """Compute extrema at one scale. Lazy-compiles numba JIT on first call."""
    global _extlis_jit
    if _extlis_jit is None:
        _extlis_jit = _compile_extrema_functions()
    return _extlis_jit(Y, dx, x0, epsilon, firstp, lastp)


def compute_extrep(coeffs, scales, n_oct, n_voice, a_min,
                   valid_ranges=None, dx=1.0, x0=0.0, epsilon=1e-6):
    """Compute extrema representation for all scales.
    Returns (extrep_abs, extrep_ord, extrep_idx) — lists of arrays per scale.

    valid_ranges: list of (firstp, lastp) from CWT, for L2 norm computation.
                  If None, uses full signal range (0, size-1).
    """
    n_scales = n_oct * n_voice
    size = coeffs.shape[1]
    extrep_abs = []
    extrep_ord = []
    extrep_idx = []
    total = 0

    for idx in range(n_scales):
        if valid_ranges is not None:
            firstp, lastp = valid_ranges[idx]
        else:
            firstp, lastp = 0, size - 1
        abs_arr, ord_arr, idx_arr = compute_extlis_numba(
            coeffs[idx], dx, x0, epsilon, firstp, lastp)
        extrep_abs.append(abs_arr)
        extrep_ord.append(ord_arr)
        extrep_idx.append(idx_arr)
        total += len(abs_arr)

    print(f'Total extrema: {total}')
    return extrep_abs, extrep_ord, extrep_idx
