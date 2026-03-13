"""Hurst exponent estimation via rescaled range (R/S) analysis."""

import numpy as np


def rescaled_range(x):
    """Compute R/S for a single series."""
    n = len(x)
    mean = np.mean(x)
    y = np.cumsum(x - mean)
    r = np.max(y) - np.min(y)
    s = np.std(x, ddof=1)
    if s == 0:
        return np.nan
    return r / s

def hurst_rs(x, min_block=8, max_blocks=40):
    """Estimate Hurst exponent via R/S analysis.
    Returns (H, se, log_n, log_rs) for plotting."""
    n = len(x)
    log_n = []
    log_rs = []

    # Use block sizes that are powers of 2 and divisors-ish
    block_sizes = []
    bs = min_block
    while bs <= n / 4:
        block_sizes.append(bs)
        bs = int(bs * 1.5)
        if len(block_sizes) >= max_blocks:
            break

    for bs in block_sizes:
        n_blocks = n // bs
        if n_blocks < 2:
            continue
        rs_vals = []
        for b in range(n_blocks):
            block = x[b * bs:(b + 1) * bs]
            rs = rescaled_range(block)
            if not np.isnan(rs) and rs > 0:
                rs_vals.append(rs)
        if len(rs_vals) > 0:
            log_n.append(np.log2(bs))
            log_rs.append(np.log2(np.mean(rs_vals)))

    log_n = np.array(log_n)
    log_rs = np.array(log_rs)

    if len(log_n) >= 3:
        H, intercept = np.polyfit(log_n, log_rs, 1)
        # Standard error of slope
        residuals = log_rs - (H * log_n + intercept)
        se = np.sqrt(np.sum(residuals**2) / (len(log_n) - 2) / np.sum((log_n - log_n.mean())**2))
    else:
        H = np.nan
        se = np.nan

    return H, se, log_n, log_rs
