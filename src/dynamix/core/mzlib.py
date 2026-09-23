import numpy as np
from dynamix.core import fftbackend

FFT = fftbackend.get_backend()   # RECON_FFT env var selects numpy (default) / pyfftw / mlx;
                                  # see fftbackend.py and 12_backends.py's agreement gates.
                                  # fftfreq (in _omega below) stays np.fft.fftfreq -- not part
                                  # of the seam (identical frequency-grid indexing everywhere).

TAPS = {
    "h": {-1: 0.125, 0: 0.375, 1: 0.375, 2: 0.125},
    "g": {0: -2.0, 1: 2.0},
    "k": {-3: 1/128, -2: 7/128, -1: 22/128, 0: -22/128, 1: -7/128, 2: -1/128},
    "l": {-3: 1/128, -2: 6/128, -1: 15/128, 0: 84/128, 1: 15/128, 2: 6/128, 3: 1/128},
}
LAMBDA_TABLE = [1.50, 1.12, 1.03, 1.01, 1.00]
THETA0 = 4.0 / 3.0  # θ(0) = 4/3 — peak of the paper's θ (eq. 102): the cubic B-spline compressed to [−1,1], θ(x) = 2·B₃(2x)

def Hf(w, n_spline=3):
    return np.exp(1j*w/2) * np.cos(w/2)**n_spline

def Gf(w):
    return 4j * np.exp(1j*w/2) * np.sin(w/2)

def Kf(w):
    G = Gf(w)
    out = np.zeros_like(G)
    nz = np.abs(G) > 1e-14
    out[nz] = (1 - np.abs(Hf(w[nz]))**2) / G[nz]
    return out                      # K(0)=0 limit: 1-|H|^2 ~ 3w^2/4, G ~ 2iw -> -i*3w/8 -> 0

def Lf_(w):
    return (1 + np.abs(Hf(w))**2) / 2

def _sinc(x):
    return np.sinc(x / np.pi)       # np.sinc(t)=sin(pi t)/(pi t); we want sin(x)/x

def phi_hat(w, n_spline=3):
    return _sinc(w/2)**n_spline

def psi_hat(w):
    return 1j * w * _sinc(w/4)**4

# === Generalized Daubechies-family filters (extension: db4/Haar continuous wavelets, script
# 16's wavelet zoo) === Same infinite-product technique as script 01's product_90_to_100 gate
# (phi_hat(w) = prod_p Hf(w/2^p)), generalized to an arbitrary FIR low-pass tap array h[0..N-1]
# (causal indexing -- Daubechies filters beyond Haar are never symmetric, so there is no
# natural center the way TAPS' h/g/k/l are centered). h must be the standard orthonormal
# convention sum(h)=sqrt(2), sum(h**2)=1; both instances below are verified against that plus
# double-shift orthogonality and vanishing moments before use (see task-ext-report.md), not
# just transcribed and trusted.
HAAR_TAPS = np.array([1.0, 1.0]) / np.sqrt(2)
DB4_TAPS = np.array([
     0.230377813308896,  0.714846570552915,  0.630880767929859, -0.027983769416859,
    -0.187034811719093,  0.030841381835560,  0.032883011666885, -0.010597401785069,
])  # standard Daubechies-4 (4 vanishing moments, 8 taps); verified: sum=sqrt(2) (1e-15),
    # sum(h^2)=1 (1e-15), double-shift orthogonality m=1,2,3 all ~1e-16, QMF companion g has 4
    # vanishing moments (~1e-14) and a nonzero 5th moment (confirms exactly 4, i.e. "db4").

def daub_hhat(w, h):
    """H-hat(w) = (1/sqrt(2)) sum_n h[n] exp(i n w) -- normalized low-pass transfer (same sign
    convention as taps_to_transfer), H-hat(0)=1 for a valid orthonormal scaling filter."""
    out = np.zeros_like(w, dtype=complex)
    for n, c in enumerate(h):
        out = out + c * np.exp(1j * n * w)
    return out / np.sqrt(2)

def daub_qmf(h):
    """Standard alternating-flip QMF high-pass companion: g[n] = (-1)^n h[N-1-n]."""
    N = len(h)
    return np.array([((-1) ** n) * h[N - 1 - n] for n in range(N)])

def daub_phi_hat(w, h, n_terms=30):
    """phi-hat via the truncated infinite product (script 01's product_90_to_100 technique,
    generalized to an arbitrary orthonormal FIR tap array): phi-hat(w) = prod_p Hhat(w/2^p)."""
    prod = np.ones_like(w, dtype=complex)
    for p in range(1, n_terms):
        prod = prod * daub_hhat(w / 2.0 ** p, h)
    return prod

def daub_psi_hat(w, h, n_terms=30):
    """psi-hat via the standard two-scale wavelet relation psi-hat(w) = Ghat(w/2)*phihat(w/2)
    (Mallat, "A Wavelet Tour of Signal Processing", ch. 7), G the QMF companion of H."""
    g = daub_qmf(h)
    return daub_hhat(w / 2.0, g) * daub_phi_hat(w / 2.0, h, n_terms)

def taps_to_transfer(taps, w):
    out = np.zeros_like(w, dtype=complex)
    for n, c in taps.items():
        out += c * np.exp(1j * n * w)
    return out

def mirror(d):
    return np.concatenate([d, d[::-1]])

def lam(j):
    return LAMBDA_TABLE[j-1] if j <= len(LAMBDA_TABLE) else 1.0

def _omega(n2):
    return 2*np.pi*np.fft.fftfreq(n2)

def atrous_forward(d, J, use_lambda=True):
    n2 = 2*len(d); w = _omega(n2)
    S = FFT.fft(mirror(d).astype(np.float64))
    Wlist = []
    for j in range(J):                      # produces detail at scale 2^{j+1}
        Wd = S * Gf((2**j) * w)
        if use_lambda: Wd = Wd / lam(j+1)   # λ indexed by the DETAIL level j+1
        Wlist.append(np.real(FFT.ifft(Wd)))
        S = S * Hf((2**j) * w)
    return np.real(FFT.ifft(S)), Wlist

def atrous_inverse(S, Wlist, use_lambda=True):
    J = len(Wlist); n2 = len(S); w = _omega(n2)
    Sh = FFT.fft(S)
    for j in range(J-1, -1, -1):
        Wd = FFT.fft(Wlist[j])
        if use_lambda: Wd = Wd * lam(j+1)
        Sh = Kf((2**j) * w) * Wd + np.conj(Hf((2**j) * w)) * Sh
    d2 = np.real(FFT.ifft(Sh))
    return d2[:n2//2]

def find_maxima(wj):
    a = np.abs(wj); left = np.roll(a, 1); right = np.roll(a, -1)
    ge = (a >= left) & (a >= right); gt = (a > left) | (a > right)
    return np.nonzero(ge & gt)[0]

def energy52(gj, scale):
    n = len(gj); w = _omega(n)
    G = FFT.fft(gj)
    return float(np.sum(np.abs(G)**2 * (1 + (scale*w)**2)) / n)

def _seg_correction(L, s, e0, e1, t):
    if L / s < 30.0:
        sh = np.sinh(L / s)
        return e0*np.sinh((L - t)/s)/sh + e1*np.sinh(t/s)/sh
    return e0*np.exp(-t/s) + e1*np.exp(-(L - t)/s)

def p_gamma(gj, idx, vals, scale):
    n = len(gj); h = gj.copy()
    if len(idx) == 0: return h
    if len(idx) == 1:
        t = np.minimum(np.abs(np.arange(n) - idx[0]), n - np.abs(np.arange(n) - idx[0]))
        return h + (vals[0] - gj[idx[0]]) * np.exp(-t / scale)
    order = np.argsort(idx); idx = np.asarray(idx)[order]; vals = np.asarray(vals)[order]
    res = vals - gj[idx]
    for m in range(len(idx)):
        a = idx[m]; b = idx[(m+1) % len(idx)] + (n if m == len(idx)-1 else 0)
        L = b - a
        if L == 0: continue
        t = np.arange(1, L)                       # interior points of the segment
        corr = _seg_correction(float(L), scale, res[m], res[(m+1) % len(idx)], t.astype(float))
        h[(a + t) % n] += corr
    h[idx % n] = vals                             # endpoints exactly
    return h

def p_v(S, Wlist):
    d2 = atrous_inverse_full(S, Wlist)
    return atrous_forward_full(d2, len(Wlist))

def atrous_forward_full(d2, J, use_lambda=True):
    n2 = len(d2); w = _omega(n2)
    Sh = FFT.fft(d2.astype(np.float64)); Wl = []
    for j in range(J):
        Wd = Sh * Gf((2**j) * w)
        if use_lambda: Wd = Wd / lam(j+1)
        Wl.append(np.real(FFT.ifft(Wd)))
        Sh = Sh * Hf((2**j) * w)
    return np.real(FFT.ifft(Sh)), Wl

def atrous_inverse_full(S, Wlist, use_lambda=True):
    J = len(Wlist); n2 = len(S); w = _omega(n2)
    Sh = FFT.fft(S)
    for j in range(J-1, -1, -1):
        Wd = FFT.fft(Wlist[j])
        if use_lambda: Wd = Wd * lam(j+1)
        Sh = Kf((2**j) * w) * Wd + np.conj(Hf((2**j) * w)) * Sh
    return np.real(FFT.ifft(Sh))

def p_y(gj, idx, vals):
    n = len(gj); h = gj.copy()
    if len(idx) < 2: return h
    order = np.argsort(idx); idx = np.asarray(idx)[order]; vals = np.asarray(vals)[order]
    for m in range(len(idx)):
        a = idx[m]; b = idx[(m+1) % len(idx)] + (n if m == len(idx)-1 else 0)
        v0, v1 = vals[m], vals[(m+1) % len(idx)]
        seg = np.arange(a, b + 1) % n
        if np.sign(v0) == np.sign(v1) and np.sign(v0) != 0:
            if v0 > 0: h[seg] = np.maximum(h[seg], 0.0)
            else:      h[seg] = np.minimum(h[seg], 0.0)
        else:
            lo, hi = min(v0, v1), max(v0, v1)
            h[seg] = np.clip(h[seg], lo, hi)
    return h

# === 2-D dyadic wavelet transform (Appendix C, D) — Task 10, the Lena stage ===
# Filter pairs confirmed against the print (md lines 936-949) before coding, per spec commit
# e08c8f5 and task brief Step 1:
#   Forward (App D):  W1_{j+1} = S_j * (G_j, D)     -- (x: G, y: Dirac)
#                      W2_{j+1} = S_j * (D, G_j)     -- (x: Dirac, y: G)
#                      S_{j+1}  = S_j * (H_j, H_j)
#   Inverse (App D):   S_{j-1} = lam_j.W1*(K_{j-1},L_{j-1}) + lam_j.W2*(L_{j-1},K_{j-1})
#                                + S_j*(Htilde_{j-1}, Htilde_{j-1})
# "A*(F1,F2)" is the paper's own convention (md line 936): F1 filters within each row (the x
# direction), F2 filters within each column (the y direction). Exactness follows from the
# same identity gated in script 01 (pr2d_identity_107_108): G(x)K(x).L(y) + L(x).G(y)K(y) +
# |H(x)|^2|H(y)|^2 = 1.
# Same lambda_j table as 1-D: Appendix D divides BOTH W1 and W2 by the SAME lambda_j (not a
# separate 2-D table) -- "We use the same notations as in Appendix B" (md line 936). Resolved
# in spec commit e08c8f5; no separate 2-D lambda derivation needed.

def atrous2d_forward(img, J, use_lambda=True):
    m = np.concatenate([img, img[::-1, :]], axis=0)
    m = np.concatenate([m, m[:, ::-1]], axis=1)
    return atrous2d_forward_full(m.astype(np.float64), J, use_lambda)

def atrous2d_forward_full(m, J, use_lambda=True):
    """Same loop as atrous2d_forward but skips the initial mirror step -- m is already on the
    (2Ny,2Nx) torus. Used by p_v2d (the P_V re-analysis half), mirroring mzlib's 1-D
    atrous_forward_full."""
    ny, nx = m.shape
    wy = _omega(ny)[:, None]; wx = _omega(nx)[None, :]
    Sh = FFT.fft2(m.astype(np.float64)); out = []
    for j in range(J):
        f1 = Gf((2**j)*wx) * np.ones_like(wy)          # rows: G, cols: Dirac  (App D)
        f2 = np.ones_like(wx) * Gf((2**j)*wy)           # rows: Dirac, cols: G
        W1 = np.real(FFT.ifft2(Sh*f1)); W2 = np.real(FFT.ifft2(Sh*f2))
        if use_lambda: W1, W2 = W1/lam(j+1), W2/lam(j+1)
        out.append((W1, W2))
        Sh = Sh * Hf((2**j)*wx) * Hf((2**j)*wy)
    return np.real(FFT.ifft2(Sh)), out

def atrous2d_inverse(S, Wpairs, use_lambda=True):
    """Full (uncropped) 2-D inverse: S may be either the (2Ny,2Nx) mirrored torus (fresh from
    atrous2d_forward) or any same-shaped intermediate produced during POCS -- this function
    never mirrors or crops; callers crop to the primary [:ny,:nx] quadrant when they want the
    physical image back."""
    J = len(Wpairs); ny, nx = S.shape
    wy = _omega(ny)[:, None]; wx = _omega(nx)[None, :]
    Sh = FFT.fft2(S)
    for j in range(J-1, -1, -1):
        W1, W2 = Wpairs[j]
        if use_lambda: W1, W2 = W1*lam(j+1), W2*lam(j+1)
        Sh = (FFT.fft2(W1)*Kf((2**j)*wx)*Lf_((2**j)*wy)
              + FFT.fft2(W2)*Lf_((2**j)*wx)*Kf((2**j)*wy)
              + Sh*np.conj(Hf((2**j)*wx))*np.conj(Hf((2**j)*wy)))
    return np.real(FFT.ifft2(Sh))

def p_v2d(S_true, Wpairs):
    """2-D P_V = W o W^-1 (eq. 64), S channel pinned to S_true (never iterated) -- the exact
    2-D analog of mzlib's 1-D p_v."""
    d2 = atrous2d_inverse(S_true, Wpairs)
    return atrous2d_forward_full(d2, len(Wpairs))

def _bilinear2d_periodic(img, r, c):
    ny, nx = img.shape
    r0 = np.floor(r).astype(int) % ny; c0 = np.floor(c).astype(int) % nx
    r1 = (r0 + 1) % ny; c1 = (c0 + 1) % nx
    dr = r - np.floor(r); dc = c - np.floor(c)
    return (img[r0, c0]*(1-dr)*(1-dc) + img[r1, c0]*dr*(1-dc)
            + img[r0, c1]*(1-dr)*dc + img[r1, c1]*dr*dc)

def nms2d_dyadic(W1, W2):
    """Gradient-direction NMS on one scale's (W1,W2) dyadic pair (eqs. 66-68): a pixel is a
    modulus maximum if M=hypot(W1,W2) is >= its two bilinear-interpolated neighbors one pixel
    along +-the (W1,W2)/M unit gradient direction, with >= on both sides and > on at least one
    (same rule as cwtlib.nms_classes' is_max). Periodic wraparound, not clip-to-edge: the
    (2Ny,2Nx) domain is the genuine mirror-periodized torus mzlib's a trous already works on
    (SIII-B), so there is no border to special-case here (unlike cwtlib's pre-mirrored-torus-fix
    Gaussian frame -- see commit 4ae9461 and spec commit 241fe76). Returns (rows, cols) index
    arrays, np.nonzero-style."""
    M = np.hypot(W1, W2)
    ny, nx = M.shape
    rr, cc = np.mgrid[0:ny, 0:nx].astype(float)
    gx = W1 / (M + 1e-30); gy = W2 / (M + 1e-30)
    mp = _bilinear2d_periodic(M, rr + gy, cc + gx)
    mm = _bilinear2d_periodic(M, rr - gy, cc - gx)
    is_max = (M >= mp) & (M >= mm) & ((M > mp) | (M > mm))
    return np.nonzero(is_max)

def pocs2d(maxima, S_true, shape, J, n_iter, checkpoint_at=None, constraint_mode="separable"):
    """SVIII POCS in 2-D. maxima[j] = (rows, cols, w1vals, w2vals): the gradient-direction
    modulus-maxima positions at scale 2^(j+1) plus BOTH dyadic components there (SVIII-A).
    NO P_Y (App E, in terms: "In two dimensions, we do not introduce any sign constraint").
    Starts from zero (min-norm limit, SVIII-A). S is pinned to S_true throughout -- the
    paper's stored coarse channel, never iterated.

    constraint_mode="separable" (default, unchanged from before this parameter existed):
    P_Gamma is applied row-wise to W1 and column-wise to W2, reusing the 1-D p_gamma exactly
    (App E: fixing y reduces the 2-D membrane problem to the 1-D x-problem of eqs. 109-113,
    and vice versa fixing x) -- the paper's own minimal-(52)-energy interpolation between
    consecutive constraints on each row/column.
    constraint_mode="set_points": dense-set fast path. Direct value assignment at the
    constraint pixels (G1[rows,cols]=w1v, G2[rows,cols]=w2v), no p_gamma interpolation and no
    row/column grouping. Exact at every constraint point either way (p_gamma is exact at its
    endpoints too); the two modes differ only in what happens to the NON-constraint pixels
    between them on the same row/column, which is exactly what interpolation is for -- so
    "set_points" is only a good approximation to "separable" when the constraint set is dense
    enough that "between consecutive constraints" is almost always "the adjacent pixel" (see
    13_wavelets.py's dense-set SNR/wall-time comparison). Not a subset of "separable"'s
    behavior at sparse constraint sets; a genuinely different (cheaper) projection.

    checkpoint_at: optional iterable of iteration counts (e.g. (10,20,30)) at which to also
    materialize the cropped image via one extra inverse transform -- lets a single n_iter=30
    run serve all three MEASURED checkpoints instead of rerunning POCS from zero three times.

    Returns (img_hat, resid, checkpoints): img_hat is the shape-cropped reconstruction after
    n_iter iterations; resid is the length-n_iter constraint-residual trajectory (sum of
    squared deviation from the true maxima values, both components, all scales); checkpoints
    is {it: cropped image at iteration it} for it in checkpoint_at (empty dict if unused)."""
    if constraint_mode not in ("separable", "set_points"):
        raise ValueError(f"unknown constraint_mode {constraint_mode!r}; "
                          "choices: 'separable', 'set_points'")
    ny, nx = shape; pny, pnx = 2*ny, 2*nx
    checkpoint_at = set(checkpoint_at or ())
    row_groups, col_groups = [], []
    if constraint_mode == "separable":
        for j in range(J):
            rows, cols, w1v, w2v = maxima[j]
            rd, cd = {}, {}
            for r, c, a, b in zip(rows, cols, w1v, w2v):
                r, c = int(r), int(c)
                rd.setdefault(r, [[], []]); rd[r][0].append(c); rd[r][1].append(a)
                cd.setdefault(c, [[], []]); cd[c][0].append(r); cd[c][1].append(b)
            row_groups.append({r: (np.array(v[0]), np.array(v[1])) for r, v in rd.items()})
            col_groups.append({c: (np.array(v[0]), np.array(v[1])) for c, v in cd.items()})

    W = [(np.zeros((pny, pnx)), np.zeros((pny, pnx))) for _ in range(J)]   # start from zero
    resid, checkpoints = [], {}
    for it in range(1, n_iter + 1):
        newW = []
        for j in range(J):
            scale = 2.0 ** (j + 1)
            G1, G2 = W[j]
            G1 = G1.copy(); G2 = G2.copy()
            if constraint_mode == "separable":
                for r, (cidx, vals) in row_groups[j].items():
                    G1[r] = p_gamma(G1[r], cidx, vals, scale)
                for c, (ridx, vals) in col_groups[j].items():
                    G2[:, c] = p_gamma(G2[:, c], ridx, vals, scale)
            else:  # set_points
                rows, cols, w1v, w2v = maxima[j]
                G1[rows, cols] = w1v
                G2[rows, cols] = w2v
            newW.append((G1, G2))
        _, W = p_v2d(S_true, newW)
        r2 = 0.0
        for j in range(J):
            rows, cols, w1v, w2v = maxima[j]
            G1, G2 = W[j]
            r2 += float(np.sum((G1[rows, cols] - w1v)**2) + np.sum((G2[rows, cols] - w2v)**2))
        resid.append(np.sqrt(r2))
        if it in checkpoint_at:
            checkpoints[it] = atrous2d_inverse(S_true, W)[:ny, :nx]
    img_hat = checkpoints.get(n_iter, atrous2d_inverse(S_true, W)[:ny, :nx])
    return img_hat, resid, checkpoints

def pocs(maxima, S_true, N, J, n_iter, use_py=False):
    n2 = 2 * N
    G = [np.zeros(n2) for _ in range(J)]        # start from zero -> min-norm limit
    resid = []
    for _ in range(n_iter):
        for j in range(J):
            idx, vals = maxima[j]
            G[j] = p_gamma(G[j], idx, vals, 2.0**(j+1))
            if use_py: G[j] = p_y(G[j], idx, vals)
        _, G = p_v(S_true, G)                   # S channel pinned to the truth
        r = sum(float(np.sum((G[j][maxima[j][0]] - maxima[j][1])**2)) for j in range(J))
        resid.append(np.sqrt(r))
    d2 = atrous_inverse_full(S_true, G)
    return d2[:N], resid
