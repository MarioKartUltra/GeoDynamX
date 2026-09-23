"""Signal catalog for NN training: generators, weight samplers, analytical formulas.

Contains ~30 synthetic signals with known multifractal spectra, spanning
monofractal (fBm, Weierstrass), deterministic multifractal (multinomial
Cantor measures), random cascades (log-normal, log-Poisson, log-gamma,
compound Poisson, IDC stable, W-cascade), MRW, and the Feigenbaum attractor.

Each signal dict has:
    'name'   : str identifier
    'signal' : ndarray, the raw signal (measure density or self-affine trace)
    'theory' : dict with 'tau_q', 'h_q', 'D_q' arrays
    'q_list' : array of q values
    'type'   : 'measure' or 'function'
                measure-type signals must be cumsum'd before WTMM
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def fbm_spectral(N, H, seed=42):
    """Fractional Brownian motion via spectral synthesis."""
    rng = np.random.default_rng(seed)
    f = np.arange(1, N // 2 + 1, dtype=float)
    amp = f ** (-(H + 0.5)) * np.exp(1j * rng.uniform(0, 2 * np.pi, N // 2))
    return np.fft.irfft(np.concatenate([[0], amp]), n=N)


def weierstrass_fn(N, H, b=2.0, n_terms=30):
    """Weierstrass-type function with Holder exponent H."""
    t = np.linspace(0, 1, N, endpoint=False)
    return sum(b ** (-n * H) * np.sin(b**n * 2 * np.pi * t + n * 0.7)
               for n in range(n_terms))


def multinomial_measure(r, p, n_levels=14, N_pts=None):
    """Deterministic multinomial cascade density on a regular grid.

    Parameters
    ----------
    r : array-like, partition ratios (sum to 1)
    p : array-like, probability weights
    n_levels : int, number of cascade levels
    N_pts : int or None, output length (default 2^n_levels)
    """
    if N_pts is None:
        N_pts = 2**n_levels
    r, p = np.array(r, float), np.array(p, float)
    cum_r = np.concatenate([[0], np.cumsum(r)])
    density = np.ones(N_pts)
    local_x = np.linspace(0, 1, N_pts, endpoint=False) + 0.5 / N_pts

    for level in range(n_levels):
        matched = np.zeros(N_pts, dtype=bool)
        for k in range(len(r)):
            mask = (local_x >= cum_r[k]) & (local_x < cum_r[k + 1]) & (density > 0)
            if p[k] > 0 and r[k] > 0:
                density[mask] *= p[k] / r[k]
                local_x[mask] = (local_x[mask] - cum_r[k]) / r[k]
                matched |= mask
            else:
                density[mask] = 0
                matched |= mask
        density[(density > 0) & ~matched] = 0
    return density


def random_cascade_batch(weight_batch_fn, n_levels=14, seed=42):
    """Random cascade with batch weight generation.

    Parameters
    ----------
    weight_batch_fn : callable(rng, n) -> array of n weights with E[W]=1
    n_levels : int
    seed : int
    """
    rng = np.random.default_rng(seed)
    N_pts = 2**n_levels
    total_weights = 2 * (N_pts - 1)
    all_weights = weight_batch_fn(rng, total_weights)

    measure = np.ones(N_pts)
    wi = 0
    for level in range(n_levels):
        block_size = N_pts >> level
        half = block_size >> 1
        for b in range(1 << level):
            start = b * block_size
            w1 = max(all_weights[wi], 1e-30)
            w2 = max(all_weights[wi + 1], 1e-30)
            wi += 2
            measure[start:start + half] *= w1
            measure[start + half:start + block_size] *= w2
    return measure


def feigenbaum_orbit(N_pts=None, n_levels=14, n_transient=10000):
    """Logistic map orbit at Feigenbaum accumulation point."""
    if N_pts is None:
        N_pts = 2**n_levels
    r_inf = 3.5699456718709449
    x = 0.5
    for _ in range(n_transient):
        x = r_inf * x * (1 - x)
    sig = np.empty(N_pts)
    for i in range(N_pts):
        x = r_inf * x * (1 - x)
        sig[i] = x
    return sig


# ---------------------------------------------------------------------------
# Weight samplers for random cascades (E[W] = 1)
# ---------------------------------------------------------------------------

def lognormal_batch(mu):
    """Log-normal weights: W = exp(mu*Z - mu^2/2), Z~N(0,1)."""
    return lambda rng, n: np.exp(mu * rng.normal(size=n) - mu**2 / 2)


def logpoisson_batch(C1, beta):
    """Log-Poisson weights: W = beta^N * exp(lam*(1-beta)), N~Poisson(lam)."""
    lam = C1 / (-np.log(beta))
    return lambda rng, n: beta**rng.poisson(lam, size=n) * np.exp(lam * (1 - beta))


def loggamma_batch(alpha):
    """Log-gamma weights: W = Gamma(alpha, 1)/alpha."""
    return lambda rng, n: rng.gamma(alpha, size=n) / alpha


def compound_poisson_batch(lam, gamma):
    """Compound Poisson weights."""
    return lambda rng, n: (gamma**rng.poisson(lam, size=n)
                           * np.exp(lam * (1 - gamma)))


def uniform_batch(a, b):
    """Uniform multiplier weights on [a_n, b_n] with E[W]=1."""
    a_n, b_n = 2 * a / (a + b), 2 * b / (a + b)
    return lambda rng, n: rng.uniform(a_n, b_n, size=n)


def stable_batch(alpha_l, C0):
    """Stable Levy weights (log-stable)."""
    from scipy.stats import levy_stable as lstable
    sigma = np.sqrt(C0)
    return lambda rng, n: np.exp(
        sigma * lstable.rvs(alpha_l, 0, size=n, random_state=rng))


# ---------------------------------------------------------------------------
# Analytical theory formulas
# ---------------------------------------------------------------------------

def cantor_tau(r, p, q_arr):
    """Solve sum(p_i^q * r_i^(-tau)) = 1 for each q.

    For multinomial measures with partition ratios r and weights p.
    Filters out zero-probability parts to handle gaps.
    """
    r, p = np.asarray(r, float), np.asarray(p, float)
    mask = p > 0
    r_nz, p_nz = r[mask], p[mask]
    tau_out = np.empty_like(q_arr, dtype=float)
    for i, qq in enumerate(q_arr):
        def eq(tau, _q=qq):
            return np.sum(p_nz ** _q * r_nz ** (-tau)) - 1.0
        tau_out[i] = brentq(eq, -50, 50)
    return tau_out


def cascade_tau(q, kind, **params):
    """Unified cascade tau(q) for random cascade models.

    All satisfy: tau(0) = -1, tau(1) = 0, concave.

    Parameters
    ----------
    q : array of moment orders
    kind : str, one of 'lognormal', 'logpoisson', 'loggamma',
           'compound_poisson', 'idc_stable'
    **params : model-specific parameters
    """
    q = np.asarray(q, float)
    if kind == "lognormal":
        C = params["mu"] ** 2 / (2 * np.log(2))
        return (q - 1) * (1 - C * q)
    elif kind == "logpoisson":
        C1, beta = params["C1"], params["beta"]
        return -1 + q + C1 / np.log(beta) * (beta**q - 1 - q * (beta - 1))
    elif kind == "loggamma":
        a = params["alpha"]
        return (q - 1) - (1 / np.log(2)) * (
            gammaln(a + q) - q * gammaln(a + 1) + (q - 1) * gammaln(a))
    elif kind == "compound_poisson":
        lam, gamma = params["lam"], params["gamma"]
        return (q - 1) - lam / np.log(2) * (gamma**q - 1 - q * (gamma - 1))
    elif kind == "idc_stable":
        alpha_l, C0 = params["alpha_l"], params["C0"]
        if abs(alpha_l - 1.0) < 1e-6:
            # Log-Cauchy limit: K(q) = C0/ln2 * q * ln(q)
            with np.errstate(invalid='ignore', divide='ignore'):
                qlnq = np.where(q > 0, q * np.log(np.maximum(q, 1e-30)), 0.0)
            return (q - 1) - C0 / np.log(2) * qlnq
        # Use np.sign(q)*|q|^alpha_l for negative q with fractional alpha
        q_pow = np.sign(q) * np.abs(q)**alpha_l
        return (q - 1) - C0 / (np.log(2) * (alpha_l - 1)) * (q_pow - q)
    elif kind == "beta_model":
        # Monofractal on-off cascade (Frisch-Sulem-Nelkin).
        # Support fraction beta, intermittency gives tau(q) = (q - 1) * (1 - log2(1/beta))
        # = (q - 1) * (1 + log2(beta)).  At beta=1 -> space-filling (H=1);
        # beta<1 -> fractal support with codimension c = -log2(beta).
        beta = params["beta"]
        return (q - 1) * (1.0 + np.log2(max(beta, 1e-12)))
    elif kind == "p_model":
        # Binomial (Meneveau-Sreenivasan) cascade with multipliers p and 1-p.
        # tau(q) = -log2(p^q + (1-p)^q).  p=0.5 -> monofractal; p->0 -> peaked.
        p = params["p"]
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return -np.log2(p**q + (1.0 - p)**q)
    elif kind == "cantor_binom":
        # Inhomogeneous binomial Cantor multinomial measure.
        # Two branches with (r, 1-r) partition ratios and (p, 1-p) weights.
        # r = 0.5, p arbitrary -> reduces to p_model (Meneveau-Sreenivasan).
        # r != 0.5 adds asymmetric support geometry (affects D_q via r).
        # Solved via cantor_tau (implicit brentq).
        r = float(min(max(params["r"], 1e-3), 1.0 - 1e-3))
        p = float(min(max(params["p"], 1e-3), 1.0 - 1e-3))
        return cantor_tau(np.array([r, 1.0 - r]),
                          np.array([p, 1.0 - p]),
                          q)
    elif kind == "cantor_gap":
        # Classical Cantor-with-gap: r_1 + r_gap + r_2 = 1, weights (p, 0, 1-p).
        # Models a multifractal on a fractal support with dust dimension
        # D_0 = log(2)/log(1/(r_1 or r_2))  (if r_1 = r_2 = r).
        r1 = float(min(max(params["r1"], 1e-3), 0.499))
        r2 = float(min(max(params["r2"], 1e-3), 0.499))
        p  = float(min(max(params["p"],  1e-3), 1.0 - 1e-3))
        rr = np.array([r1, max(1.0 - r1 - r2, 1e-6), r2])
        pp = np.array([p, 0.0, 1.0 - p])
        return cantor_tau(rr, pp, q)
    else:
        raise ValueError(f"Unknown cascade kind: {kind!r}")


def legendre(tau, q):
    """Legendre transform: h = dtau/dq, D = q*h - tau.

    Handles non-finite tau values gracefully.
    """
    with np.errstate(invalid='ignore'):
        h = np.gradient(tau, q)
        D = q * h - tau
    return h, D


def _monofractal_theory(H, q_arr):
    """Analytical spectra for monofractal signal with Holder exponent H."""
    tau = H * q_arr - 1.0
    h = np.full_like(q_arr, H)
    D = np.full_like(q_arr, 1.0)
    return {'tau_q': tau, 'h_q': h, 'D_q': D}


def _devil_staircase_theory(r, p, q_arr, expo=-1.0):
    """Theory for WTMM on cumsum'd multinomial measure (devil's staircase).

    The CWT with expo normalization gives:
        h_eff = h_measure + 1 + expo
        tau_eff = tau_measure + (1 + expo) * q

    With expo=-1 (default): tau_eff = tau_measure (cancellation).
    """
    tau_measure = cantor_tau(r, p, q_arr)
    tau_wtmm = tau_measure + (1.0 + expo) * q_arr
    h, D = legendre(tau_wtmm, q_arr)
    return {'tau_q': tau_wtmm, 'h_q': h, 'D_q': D}


def _cascade_theory(q_arr, kind, expo=-1.0, **params):
    """Theory for WTMM on cumsum'd random cascade.

    Same expo correction as devil's staircase: cumsum adds +1 to h,
    expo normalization adds expo to h, so tau_eff = tau_cascade + (1+expo)*q.
    """
    tau_cascade = cascade_tau(q_arr, kind, **params)
    tau_wtmm = tau_cascade + (1.0 + expo) * q_arr
    h, D = legendre(tau_wtmm, q_arr)
    return {'tau_q': tau_wtmm, 'h_q': h, 'D_q': D}


# ---------------------------------------------------------------------------
# Full signal catalog builder
# ---------------------------------------------------------------------------

def build_signal_catalog(q_arr=None, n_levels=14, expo=-1.0, seed=42):
    """Build the full catalog of ~30 synthetic signals with analytical spectra.

    Parameters
    ----------
    q_arr : array of q values (default: adaptive_q_grid())
    n_levels : int, cascade depth (signal length = 2^n_levels)
    expo : float, CWT normalization exponent (default -1.0 for LastWave)
    seed : int, RNG seed for random cascades

    Returns
    -------
    catalog : list of dict, each with keys:
        'name', 'signal', 'theory', 'q_list', 'type'
    """
    if q_arr is None:
        from .datasets import adaptive_q_grid
        q_arr = adaptive_q_grid()

    N = 2**n_levels
    catalog = []
    rng_offset = 0  # offset seeds for different cascade instances

    # --- Monofractal: fBm ---
    for H in [0.3, 0.5, 0.7, 0.9]:
        catalog.append({
            'name': f'fBm_H{H}',
            'signal': fbm_spectral(N, H, seed=seed + rng_offset),
            'theory': _monofractal_theory(H, q_arr),
            'q_list': q_arr,
            'type': 'function',
        })
        rng_offset += 1

    # --- Monofractal: Weierstrass ---
    for H in [0.5, 0.8]:
        catalog.append({
            'name': f'Weierstrass_H{H}',
            'signal': weierstrass_fn(N, H),
            'theory': _monofractal_theory(H, q_arr),
            'q_list': q_arr,
            'type': 'function',
        })

    # --- Deterministic Cantor measures (integrated before WTMM) ---
    cantor_configs = [
        ('Cantor_binom_sym', [0.5, 0.5], [0.5, 0.5]),
        ('Cantor_binom_asym', [0.5, 0.5], [0.7, 0.3]),
        ('Cantor_trinom', [0.3, 0.2, 0.5], [0.2, 0.5, 0.3]),
        ('Cantor_trinom_gap', [0.4, 0.2, 0.4], [0.6, 0.0, 0.4]),
        ('Baker_area', [0.3, 0.7], [0.3, 0.7]),
        ('Baker_nonunif', [0.3, 0.7], [0.5, 0.5]),
        ('Smale_horseshoe', [1/3, 1/3], [0.6, 0.4]),
    ]
    for name, r, p in cantor_configs:
        sig = multinomial_measure(r, p, n_levels=n_levels)
        theory = _devil_staircase_theory(r, p, q_arr, expo=expo)
        catalog.append({
            'name': name,
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- Random cascades: log-normal ---
    for mu in [0.2, 0.4, 0.6]:
        rng_offset += 1
        sig = random_cascade_batch(lognormal_batch(mu),
                                   n_levels=n_levels, seed=seed + rng_offset)
        theory = _cascade_theory(q_arr, 'lognormal', expo=expo, mu=mu)
        catalog.append({
            'name': f'LogNormal_mu{mu}',
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- Random cascades: log-Poisson ---
    for C1, beta in [(0.15, 0.5), (0.3, 0.3)]:
        rng_offset += 1
        sig = random_cascade_batch(logpoisson_batch(C1, beta),
                                   n_levels=n_levels, seed=seed + rng_offset)
        theory = _cascade_theory(q_arr, 'logpoisson', expo=expo,
                                 C1=C1, beta=beta)
        catalog.append({
            'name': f'LogPoisson_C1{C1}_b{beta}',
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- Random cascades: log-gamma ---
    for alpha in [2.0, 5.0]:
        rng_offset += 1
        sig = random_cascade_batch(loggamma_batch(alpha),
                                   n_levels=n_levels, seed=seed + rng_offset)
        theory = _cascade_theory(q_arr, 'loggamma', expo=expo, alpha=alpha)
        catalog.append({
            'name': f'LogGamma_a{alpha}',
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- MRW (same tau as log-normal, parameterized by lambda^2) ---
    for lam2 in [0.02, 0.05, 0.1]:
        rng_offset += 1
        mu_equiv = np.sqrt(2 * lam2 * np.log(2))
        sig = random_cascade_batch(lognormal_batch(mu_equiv),
                                   n_levels=n_levels, seed=seed + rng_offset)
        theory = _cascade_theory(q_arr, 'lognormal', expo=expo, mu=mu_equiv)
        catalog.append({
            'name': f'MRW_lam2_{lam2}',
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- Compound Poisson cascades ---
    for lam, gamma in [(2.0, 0.5), (1.0, 0.3)]:
        rng_offset += 1
        sig = random_cascade_batch(compound_poisson_batch(lam, gamma),
                                   n_levels=n_levels, seed=seed + rng_offset)
        theory = _cascade_theory(q_arr, 'compound_poisson', expo=expo,
                                 lam=lam, gamma=gamma)
        catalog.append({
            'name': f'CompPoisson_l{lam}_g{gamma}',
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- W-cascade (uniform multipliers) ---
    for a, b in [(0.2, 1.8), (0.5, 1.5)]:
        rng_offset += 1
        sig = random_cascade_batch(uniform_batch(a, b),
                                   n_levels=n_levels, seed=seed + rng_offset)
        # W-cascade tau: E[W^q] = (b_n^{q+1} - a_n^{q+1}) / ((q+1)*(b_n-a_n))
        a_n = 2 * a / (a + b)
        b_n = 2 * b / (a + b)
        # Handle q=-1 singularity with L'Hopital limit
        with np.errstate(divide='ignore', invalid='ignore'):
            EWq = (b_n**(q_arr + 1) - a_n**(q_arr + 1)) / (
                (q_arr + 1) * (b_n - a_n))
        # At q=-1: E[W^q] -> (ln(b_n) - ln(a_n)) / (b_n - a_n)
        singular = ~np.isfinite(EWq)
        if np.any(singular):
            EWq[singular] = (np.log(b_n) - np.log(a_n)) / (b_n - a_n)
        tau_wc = -(1 + np.log2(EWq))
        # Apply expo correction for cumsum'd measure
        tau_wtmm = tau_wc + (1.0 + expo) * q_arr
        h_wc, D_wc = legendre(tau_wtmm, q_arr)
        catalog.append({
            'name': f'Wcascade_{a}_{b}',
            'signal': sig,
            'theory': {'tau_q': tau_wtmm, 'h_q': h_wc, 'D_q': D_wc},
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- IDC stable cascades ---
    for alpha_l, C0 in [(1.5, 0.05), (1.8, 0.03)]:
        rng_offset += 1
        sig = random_cascade_batch(stable_batch(alpha_l, C0),
                                   n_levels=n_levels, seed=seed + rng_offset)
        theory = _cascade_theory(q_arr, 'idc_stable', expo=expo,
                                 alpha_l=alpha_l, C0=C0)
        catalog.append({
            'name': f'IDCstable_a{alpha_l}_C{C0}',
            'signal': sig,
            'theory': theory,
            'q_list': q_arr,
            'type': 'measure',
        })

    # --- Feigenbaum attractor (function, NOT integrated) ---
    alpha_F = 2.502907875095892
    r_feig = [1 / alpha_F**2, 1 / alpha_F]
    d_feig = 0.538
    p_feig_raw = [r_feig[0]**d_feig, r_feig[1]**d_feig]
    p_feig = [pi / sum(p_feig_raw) for pi in p_feig_raw]
    tau_feig = cantor_tau(np.array(r_feig), np.array(p_feig), q_arr)
    h_feig, D_feig = legendre(tau_feig, q_arr)
    catalog.append({
        'name': 'Feigenbaum',
        'signal': feigenbaum_orbit(N_pts=N, n_levels=n_levels),
        'theory': {'tau_q': tau_feig, 'h_q': h_feig, 'D_q': D_feig},
        'q_list': q_arr,
        'type': 'function',
    })

    return catalog
