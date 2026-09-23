# WTMM 1D — Wavelet Transform Modulus Maxima Package

A Python implementation of the WTMM method for multifractal analysis, faithfully ported from LastWave 3 with additional support for Mathematica-style scaling conventions. Designed for geophysical time series analysis (creepmeter, turbulence, seismology) with multiple CWT backends (PyTorch GPU, FFTW CPU, MLX Apple Silicon).

---

## Package Structure

```
wtmm/
├── __init__.py          # Lazy-loaded exports
├── wavelets.py          # Wavelet definitions (Gaussian derivatives, B-splines, etc.)
├── cwt.py               # CWT engines + auto-scaling + scale table
├── extrema.py           # WTMM extrema detection (numba)
├── chains.py            # Maxima line chaining across scales (numba)
├── partition.py         # Partition function (7 arrays) + ensemble variance
├── spectra.py           # Multifractal spectra: canonical & microcanonical
├── stats.py             # Wavelet skewness, flatness, correlations
├── colormaps.py         # Scalogram visualization (LastWave colormaps)
├── signals.py           # Test signals (Cantor measure, etc.)
├── hurst.py             # Rescaled range (R/S) Hurst exponent
└── io.py                # Data I/O for USGS creepmeter files
```

---

## Quick Start

```python
import numpy as np
import wtmm

# 1. Generate or load a signal
signal, dx = wtmm.ucantor(2**17, r=[0.5, 0.5], p=[0.7, 0.3])

# 2. Auto-compute scale parameters
params = wtmm.auto_scale_params(len(signal), wavelet_name='g2', method='lw')
a_min, n_oct, n_voice = params['a_min'], params['n_oct'], params['n_voice']

# 3. Compute CWT
scales = wtmm.compute_scales(a_min, n_oct, n_voice)
cwt_matrix, causal = wtmm.cwtd(signal, a_min, n_oct, n_voice, 'g2', border='mir')

# 4. Find WTMM skeleton
extrep_pos, extrep_ord = wtmm.compute_extrep(cwt_matrix, causal)
links = wtmm.chain_all(extrep_pos, extrep_ord, n_oct, n_voice)

# 5. Compute partition function
q_list = np.arange(-3, 6.1, 0.1)
pf = wtmm.compute_partition_function(extrep_ord, n_oct, n_voice, a_min, q_list)

# 6. Compute multifractal spectra
sp = wtmm.compute_spectra(pf, log2_a_min=1.0, log2_a_max=6.0, method='canonical')
```

---

## Module Reference

### `cwt.py` — Continuous Wavelet Transform

#### `auto_scale_params(signal_size, n_voice=None, wavelet_name='g2', method='lw')`

Compute optimal scale parameters from signal length and wavelet.

| Method | `a_min` | `n_oct` | Default `n_voice` |
|--------|---------|---------|-------------------|
| `'lw'` (LastWave) | 1.0 (hardcoded) | From wavelet support fitting in signal | 5 |
| `'math'` (Mathematica) | α_scale / fact (from wavelet peak frequency) | floor(log₂(N/2)) | 4 |

**Returns:** dict with `a_min`, `a_max`, `n_oct`, `n_voice`, `method`

#### `scale_table(a_min, n_oct, n_voice, wavelet_name='g2', dx=1.0, units='samples')`

Print a table showing octave, voice, scale index, scale parameter `a`, and physical scale (`a × fact × dx`) for each voice.

#### `compute_scales(a_min, n_oct, n_voice)`

Return array of scale values: `a_min * 2^(i/n_voice)` for i = 0 .. n_oct*n_voice - 1.

#### `cwtd(signal, a_min, n_oct, n_voice, wavelet_name, border='mir', expo=-1)`

Compute the CWT using direct convolution (NumPy/PyTorch).

- `border`: `'mir'` (mirror), `'per'` (periodic), `'pad'` (zero-pad)
- `expo`: normalization exponent. Use -1 for WTMM (default), 0 for spectral analysis.

**Returns:** `(cwt_matrix, causal_ranges)` — shape (n_scales, n_samples) and list of (first, last) valid sample ranges per scale.

#### `cwtd_fftw(signal, ...)` / `cwtd_mlx(signal, ...)`

FFT-based backends using PyFFTW (CPU) or MLX (Apple Silicon GPU).

---

### `partition.py` — Partition Functions

#### Seven Arrays per (q, scale)

All computed per-scale by `pf_compute_one_scale` (numba JIT), matching LastWave's `PFComputeOneScaleFLOAT`:

| Array | Formula | Role |
|-------|---------|------|
| `sTq` | Σ\|T\|^q | Z(q,a) — raw partition sum |
| `sTqLogT` | Σ\|T\|^q · ln\|T\| | Numerator of H(q,a) |
| `logSTq` | ln(Z/N) | Intensive log partition function |
| `sTqLogT_sTq` | sTqLogT / sTq | H(q,a) — Boltzmann-weighted avg of ln\|T\| |
| `log2STq` | (logSTq)² | For ensemble variance of T |
| `sTqLogT_sTq2` | (sTqLogT_sTq)² | For ensemble variance of H |
| `logSTqSTqLogT_sTq` | logSTq × sTqLogT_sTq | For ensemble variance of D |

#### `compute_partition_function(extrep_ord, n_oct, n_voice, a_min, q_list, ...)`

Main entry point. Optionally filters by chain length (`min_chain_voices`).

**Returns:** dict with all 7 arrays (shape: n_q × n_scales), plus metadata (`q_list`, `scales`, `log2_a`, `n_ext`, `index_max`, `signal_number`).

#### Accessors

| Function | Returns | Formula |
|----------|---------|---------|
| `pf_get_T(pf, q_idx, mode)` | T(q,a) = log₂ Z(q,a) | log partition function |
| `pf_get_H(pf, q_idx, mode)` | H(q,a) = sTqLogT_sTq / ln(2) | Boltzmann-weighted avg of log\|T\| |
| `pf_get_D(pf, q_idx, q_val, mode)` | D(q,a) = q·H - T | Boltzmann entropy |

- `mode='extensive'`: single signal (uses raw sTq)
- `mode='intensive'`: ensemble average (uses logSTq/N)

#### Ensemble Functions

| Function | Purpose |
|----------|---------|
| `pf_standard_addition(pf_accum, pf_new)` | Accumulate partition functions from multiple signals (in-place) |
| `pf_copy(pf)` | Deep copy a partition function dict |
| `pf_get_var_T(pf, q_idx)` | Variance of T across ensemble (requires signal_number ≥ 2) |
| `pf_get_var_H(pf, q_idx)` | Variance of H across ensemble |
| `pf_get_var_D(pf, q_idx, q_val)` | Variance of D across ensemble |

Ensemble workflow:
```python
pf_total = wtmm.pf_copy(pf_segment_1)
for pf_seg in pf_segments[1:]:
    wtmm.pf_standard_addition(pf_total, pf_seg)
# Now pf_total['signal_number'] == len(pf_segments)
var_h = wtmm.pf_get_var_H(pf_total, q_idx=5)  # ensemble error bars
```

---

### `spectra.py` — Multifractal Spectra

#### `compute_spectra(pf, log2_a_min, log2_a_max, mode='extensive', method='canonical', L_ref=None)`

Compute τ(q), h(q), D(q) via linear regression of partition function quantities vs log₂(a).

##### Two Methods

**Canonical** (recommended):
- h(q) = slope of H(q,a) vs log₂(a) — direct fit
- D(q) = slope of D(q,a) vs log₂(a) — direct fit
- τ(q) = q·h - D (derived)
- Preserves non-convex D(h) and phase transitions
- q = 1 is well-defined (no 0/0 singularity)

**Microcanonical** (alias: `'legendre'`):
- τ(q) = slope of T(q,a) vs log₂(a) — direct fit
- h(q) = dτ/dq (numerical derivative)
- D(q) = q·h - τ (Legendre transform)
- Gives convex hull of D(h)
- Automatically normalizes by τ(1) for smooth Rényi dimensions at q = 1
- Returns both true values (corrected) and normalized values

##### Output Dict

| Key | Both methods | Description |
|-----|-------------|-------------|
| `tau_q` | yes | Scaling exponents (true/unnormalized) |
| `h_q` | yes | Hölder exponents |
| `D_q` | yes | Fractal dimensions (true/unnormalized) |
| `q_list` | yes | q values |
| `tau_1` | yes | τ(1) value |
| `method` | yes | `'canonical'` or `'microcanonical'` |
| `h_err` | yes | Standard error on h(q) from regression |
| `D_err` | yes | Standard error on D(q) from regression |
| `tau_err` | yes | Standard error on τ(q) from regression |
| `h_R2` | yes | R² of h(q,a) fit (NaN for microcanonical) |
| `D_R2` | yes | R² of D(q,a) fit (NaN for microcanonical) |
| `tau_R2` | yes | R² of T(q,a) fit |
| `L_ref` | yes | Reference scale (None if not set) |
| `log2_a_norm` | if L_ref set | Normalized scale axis: log₂(a/L_ref) |
| `tau_q_norm` | microcanonical | Normalized τ with τ_norm(1) = 0 |
| `D_q_norm` | microcanonical | Normalized D (smooth Rényi dims) |

##### Rényi Dimensions

```python
# From microcanonical method:
D_q_smooth = sp['tau_q_norm'] / (sp['q_list'] - 1)  # smooth at q=1
D_q_true = sp['tau_q'] / (sp['q_list'] - 1)         # blows up at q=1

# D_1 (information dimension) — use h(1) directly:
q1_idx = np.argmin(np.abs(sp['q_list'] - 1.0))
D_1 = sp['h_q'][q1_idx]
```

##### Reference Scale Normalization

```python
# Compare instruments with different characteristic scales:
sp = wtmm.compute_spectra(pf, log2_a_min=-4.0, log2_a_max=0.5, L_ref=100.0)
# log2_a_min=-4 means a/L_ref = 1/16 (scales 16× smaller than L_ref)
# Plot partition functions vs sp['log2_a_norm'] for cross-instrument comparison
```

#### `theoretical_devil_staircase(q_array, r, p, expo=-1.0)`

Compute theoretical τ(q), h(q), D(h) for the WTMM analysis of a devil's staircase (integral of a self-similar measure).

---

### `stats.py` — Wavelet Statistical Diagnostics

These operate on the raw CWT matrix, not the WTMM skeleton.

#### `wavelet_skewness_flatness(cwt_matrix, causal_ranges=None)`

Per-scale skewness and flatness (kurtosis) of CWT coefficients.

- Skewness(a) = ⟨T³⟩ / ⟨T²⟩^{3/2} — asymmetry (e.g. ramp-cliff structures)
- Flatness(a) = ⟨T⁴⟩ / ⟨T²⟩² — heavy tails / intermittency
- Gaussian signal: skewness = 0, flatness = 3 at all scales
- Flatness >> 3 at small scales indicates intermittency

**Returns:** dict with `skewness`, `flatness`, `variance` arrays (n_scales,).

```python
sf = wtmm.wavelet_skewness_flatness(cwt_matrix, causal_ranges=causal)
plt.semilogy(log2_a, sf['flatness'], label='Flatness')
plt.axhline(3.0, ls='--', label='Gaussian')
```

#### `magnitude_correlation_cross_scale(cwt_matrix, fine_scale_idx, causal_ranges=None)`

Cross-scale correlation of wavelet modulus (Dupont et al. 2020, Fig. 12). Pearson correlation between |T(t, a_fine)| and |T(t, a)| for each scale a.

Answers: "Are fine-scale fluctuations driven by larger-scale structures?"

**Returns:** 1D array (n_scales,) of correlation values.

```python
corr = wtmm.magnitude_correlation_cross_scale(cwt_matrix, fine_scale_idx=5)
plt.plot(log2_a, corr)
plt.xlabel('log₂(a)')
plt.ylabel('Correlation with fine scale')
```

#### `space_scale_correlation(cwt_matrix, scale_idx_1, scale_idx_2=None, max_lag=None, causal_ranges=None)`

Two-point space-scale magnitude correlation (Arneodo et al. 1998).

C(Δx, a₁, a₂) = ⟨ω̃(x, a₁) · ω̃(x+Δx, a₂)⟩

where ω = ln|T| (log-magnitude), ω̃ = ω - ⟨ω⟩.

- One-scale: `scale_idx_2=None` (a₂ = a₁)
- Two-scale: provide both indices
- Log-linear decay of C vs log₂(Δx) = signature of multiplicative cascade

**Returns:** dict with `lags`, `C`, `C_norm` (normalized by C(0)).

```python
result = wtmm.space_scale_correlation(cwt_matrix, scale_idx_1=10, max_lag=5000)
plt.semilogx(result['lags'], result['C_norm'])
plt.xlabel('Δx (samples)')
plt.ylabel('C(Δx) / C(0)')
```

---

### `colormaps.py` — Scalogram Visualization

#### `scalogram_lastwave(cwt_matrix, scales=None, ax=None, ...)`

Plot a CWT scalogram using LastWave's rendering approach.

Key parameters:
- `norm_mode`: `'lglobal'` (per-scale), `'global'`, `'signedglobal'`
- `signed`: preserve sign of coefficients (uses blue-red colormap)
- `causal_ranges`: mask out border-effect samples
- `x_mode`: `'index'` or `'units'` (multiply by dx)
- `y_mode`: `'log2a'`, `'octave'`, `'voice'`, `'units'` (physical scale = a × fact × dx)
- `dx`, `units`: sample spacing and unit label
- `n_voice`: required for `'octave'` and `'voice'` y-modes
- `max_ticks`: prevent tick label overlap (default 12)

```python
wtmm.scalogram_lastwave(
    cwt_matrix, scales=scales, causal_ranges=causal,
    y_mode='units', x_mode='units',
    dx=600.0, units='seconds', n_voice=5,
    wavelet_name='g2'
)
```

---

### `wavelets.py` — Wavelet Definitions

The `WAVELETS` dict contains wavelet functions and metadata:

| Key | Type | Description |
|-----|------|-------------|
| `func` | callable | ψ(u) where u = x/(a·fact) |
| `fact` | float | Stretch factor (physical scale = a × fact) |
| `x_min_factor` | float | Left support bound (in units of fact) |
| `x_max_factor` | float | Right support bound (in units of fact) |

Available wavelets include: `g1`–`g6` (Gaussian derivatives), `bspline1`–`bspline6`, fractional B-splines, q-Gaussian, q-Mexican hat, cascade wavelets.

---

### `signals.py` — Test Signals

#### `ucantor(size, r, p, nFlip=0)`

Generate non-uniform Cantor measure (for validating against theoretical D(h)).

#### `tau_equation(q, r, p)`

Solve the theoretical τ(q) equation for self-similar measures: Σ p_i^q · r_i^{-τ} = 1.

---

### `hurst.py` — Hurst Exponent

#### `hurst_rs(x, min_block=8, max_blocks=40)`

Estimate Hurst exponent via rescaled range (R/S) analysis. Returns (H, se, log_n, log_rs).

---

## Theoretical Background

### Partition Function Quantities

At each scale a, from the WTMM skeleton maxima |T_ℓ|:

- **Z(q,a)** = Σ |T_ℓ|^q — partition sum (our `sTq`)
- **Boltzmann weight**: Ŵ(q,ℓ,a) = |T_ℓ|^q / Z(q,a)
- **H(q,a)** = Σ ln|T_ℓ| · Ŵ — weighted average of log-amplitude (our `sTqLogT_sTq`)
- **D(q,a)** = Σ Ŵ · ln Ŵ = q·H - ln Z — Boltzmann entropy

### Scaling Relations

In the scaling regime, these quantities are linear in log₂(a):

- T(q,a) = log₂ Z ~ τ(q) · log₂(a)
- H(q,a) ~ h(q) · log₂(a)
- D(q,a) ~ D(q) · log₂(a)

The slopes give the multifractal spectrum parametrized by q: {h(q), D(q)}.

### Canonical vs Microcanonical

| | Canonical | Microcanonical |
|---|---|---|
| Fits | H(q,a) and D(q,a) slopes | T(q,a) slope |
| Derives | τ = qh - D | h = dτ/dq, D = qh - τ |
| D(h) shape | Can be non-convex | Always convex (Legendre hull) |
| Phase transitions | Visible | Smoothed away |
| q = 1 | Well-defined | Requires τ(1) normalization |

### Expo Convention

The CWT normalization exponent affects singularity measurement:
- **expo = -1** (L1, default): |T| ~ a^h where h is the Hölder exponent directly. Always use for WTMM.
- **expo = 0** (L2): energy-preserving, for spectral analysis.

### Log-Normal Model

A quadratic fit to τ(q) gives a 3-parameter summary of multifractality:

τ(q) ≈ -c₀ + c₁·q - c₂·q²/2

| Coefficient | Physical meaning |
|-------------|-----------------|
| c₀ = -τ(0) | Codimension of support (≈ D₀) |
| c₁ | Most probable singularity h₀ |
| c₂ | Intermittency coefficient (width of D(h)) |

c₂ = 0 → monofractal. Larger c₂ → wider D(h) → more multifractal.

### Phase Transitions

When τ(q) saturates at τ_sat for q > q_crit, this indicates a phase transition: strong singularities (h ≈ 0, e.g. cliffs/jumps) dominate the high-q moments. Detectable by:
1. τ(q) flattening above q_crit
2. Linear left tail in D(h) dropping to D(h=0) = -τ_sat
3. Tangent line with slope q_crit connecting the multifractal D(h) to D(h=0)

---

## Dependencies

**Required:** numpy, matplotlib, scipy

**Optional (lazy-imported):**
- `numba` — JIT compilation for partition functions and extrema detection
- `torch` — GPU-accelerated CWT (`cwtd` with CUDA)
- `pyfftw` — FFT-based CWT (`cwtd_fftw`)
- `mlx` — Apple Silicon GPU CWT (`cwtd_mlx`)
- `pandas` — Data I/O for creepmeter files

---

## References

- Muzy, Bacry & Arneodo (1991, 1993, 1994) — WTMM method foundations
- Arneodo, Bacry & Muzy (1995) — Canonical method, τ(1) normalization
- Arneodo, Bacry, Manneville & Muzy (1998) — Space-scale correlation functions
- Dupont, Argoul, Gerasimova-Chechkina, Irvine & Arneodo (2020) — Phase transitions in multifractal spectra, wavelet skewness/flatness
- LastWave 3 source code: `pf_lib.c`, `cwt1d.c`, `wt1d_collection.c`
