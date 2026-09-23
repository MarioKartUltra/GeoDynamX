"""MLX neural networks for robust partition function slope estimation.

Two architectures:
1. SlopeEstimator (v1): Direct prediction of spectra from partition function curves.
2. ScaleWeightEstimator (v2): Two-headed model that learns optimal scale fitting
   ranges via differentiable weighted OLS, plus a direct spectra predictor.

Input channels: T(q,a), H(q,a), D(q,a), C(q,a), [n_ext(a)] — where C is the
empirical specific heat computed from T via second derivative along q.
"""

import mlx.core as mx
import mlx.nn as nn


class ScaleConvBlock(nn.Module):
    """1D convolution along the scale axis for each q independently.

    Input:  (batch, channels, n_q, n_scales)
    Output: (batch, out_channels, n_q, n_scales')
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)

    def __call__(self, x):
        # x: (B, C, Q, S)
        B, C, Q, S = x.shape
        x = x.transpose(0, 2, 1, 3)  # (B, Q, C, S)
        x = x.reshape(B * Q, C, S)
        # Conv1d expects (N, S, C) in MLX
        x = x.transpose(0, 2, 1)  # (B*Q, S, C)
        x = self.conv(x)  # (B*Q, S', out_C)
        x = self.norm(x)
        x = nn.gelu(x)
        S_out = x.shape[1]
        out_C = x.shape[2]
        x = x.transpose(0, 2, 1)  # (B*Q, out_C, S')
        x = x.reshape(B, Q, out_C, S_out)
        x = x.transpose(0, 2, 1, 3)  # (B, out_C, Q, S')
        return x


class SlopeEstimator(nn.Module):
    """Neural network for estimating tau(q), h(q), D(q), C(q) slopes and
    detecting phase transitions from partition function curves.

    Parameters
    ----------
    n_q : int
        Number of q values in the partition function.
    n_scales : int
        Number of scales in the partition function.
    n_meta : int
        Number of metadata features (default 13).
    hidden_dim : int
        Hidden layer width for the MLP head.
    n_conv_channels : int
        Number of channels in convolutional layers.
    n_input_channels : int
        Number of input curve channels (default 4: T, H, D, C).
    """

    def __init__(self, n_q=47, n_scales=50, n_meta=13,
                 hidden_dim=256, n_conv_channels=32, n_input_channels=4):
        super().__init__()
        self.n_q = n_q
        self.n_scales = n_scales
        self.n_input_channels = n_input_channels

        # Convolutional feature extractor along scale axis
        self.conv1 = ScaleConvBlock(n_input_channels, n_conv_channels, kernel_size=7)
        self.conv2 = ScaleConvBlock(n_conv_channels, n_conv_channels, kernel_size=5)
        self.conv3 = ScaleConvBlock(n_conv_channels, n_conv_channels, kernel_size=3)

        # Direct linear slope estimate per q
        self.slope_proj = nn.Linear(n_scales, 4)  # 4 features per q

        # Shared backbone
        backbone_dim = n_conv_channels * n_q + 4 * n_q + n_meta
        self.backbone = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Spectra head: tau(q), h(q), D(q)
        self.spectra_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3 * n_q),
        )

        # Specific heat head: C(q)
        self.heat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_q),
        )

        # Phase transition head: [has_transition (logit), q_star]
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def _extract_features(self, x_curves, x_meta):
        """Shared feature extraction."""
        B = x_curves.shape[0]
        n_ch = x_curves.shape[1]

        # Conv feature extraction
        h = self.conv1(x_curves)
        h = self.conv2(h)
        h = self.conv3(h)

        # Global average pool over scales
        h = mx.mean(h, axis=3)  # (B, C, Q)
        h = h.reshape(B, -1)  # (B, C*Q)

        # Direct slope features
        curves_flat = x_curves.reshape(B * n_ch * self.n_q, self.n_scales)
        slope_feat = self.slope_proj(curves_flat)  # (B*ch*Q, 4)
        slope_feat = slope_feat.reshape(B, n_ch, self.n_q, 4)
        slope_feat = mx.mean(slope_feat, axis=1)  # (B, Q, 4)
        slope_feat = slope_feat.reshape(B, -1)  # (B, Q*4)

        features = mx.concatenate([h, slope_feat, x_meta], axis=1)
        return self.backbone(features)

    def __call__(self, x_curves, x_meta):
        """Forward pass.

        Parameters
        ----------
        x_curves : (B, 4, n_q, n_scales) — T, H, D, C curves
        x_meta   : (B, n_meta) — encoded metadata

        Returns
        -------
        spectra    : (B, 3, n_q) — predicted [tau(q), h(q), D(q)]
        heat       : (B, n_q) — predicted C(q)
        transition : (B, 2) — [has_transition logit, q_star]
        """
        B = x_curves.shape[0]
        features = self._extract_features(x_curves, x_meta)

        spectra = self.spectra_head(features).reshape(B, 3, self.n_q)
        heat = self.heat_head(features)
        transition = self.transition_head(features)

        return spectra, heat, transition

    def predict(self, x_curves, x_meta):
        """Convenience: returns numpy dicts."""
        import numpy as np
        spectra, heat, transition = self(x_curves, x_meta)
        s = np.array(spectra)
        h = np.array(heat)
        t = np.array(transition)
        return {
            'tau_q': s[:, 0, :],
            'h_q': s[:, 1, :],
            'D_q': s[:, 2, :],
            'C_q': h,
            'has_transition': t[:, 0] > 0,  # sigmoid threshold
            'q_star': t[:, 1],
        }


def compute_loss_with_q(model, x_curves, x_meta, y_true, q_array,
                        y_C=None, y_trans=None, q_mask=None,
                        consistency_weight=0.1, width_weight=0.1,
                        heat_weight=0.05, transition_weight=0.1):
    """Loss with q masking, specific heat, and phase transition terms.

    Parameters
    ----------
    y_true : (B, 3, n_q) — [tau, h, D] ground truth
    q_array : (n_q,) array of q values
    y_C : (B, n_q) — analytical C(q), optional
    y_trans : (B, 2) — [has_transition (0/1), q_star], optional
    q_mask : (B, n_q) — 1.0 where valid, 0.0 where divergent
    """
    spectra, heat_pred, trans_pred = model(x_curves, x_meta)

    tau_pred = spectra[:, 0, :]
    h_pred = spectra[:, 1, :]
    D_pred = spectra[:, 2, :]

    tau_true = y_true[:, 0, :]
    h_true = y_true[:, 1, :]

    # --- Masked spectra MSE ---
    if q_mask is not None:
        mask = mx.expand_dims(q_mask, axis=1)  # (B, 1, n_q)
        n_valid = mx.maximum(mx.sum(q_mask) * 3.0, mx.array(1.0))
    else:
        mask = mx.ones_like(y_true)
        n_valid = mx.array(float(y_true.size))

    sq_err = (spectra - y_true) ** 2 * mask
    mse = mx.sum(sq_err) / n_valid

    # --- Consistency: tau = q*h - D ---
    q_bc = mx.expand_dims(q_array, axis=0)
    tau_implied = q_bc * h_pred - D_pred
    cons_err = (tau_pred - tau_implied) ** 2
    if q_mask is not None:
        cons_err = cons_err * q_mask
        consistency = mx.sum(cons_err) / mx.maximum(mx.sum(q_mask), mx.array(1.0))
    else:
        consistency = mx.mean(cons_err)

    # --- Spectrum width ---
    if q_mask is not None:
        INF = mx.array(1e6)
        h_true_max = mx.where(q_mask > 0.5, h_true, -INF)
        h_true_min = mx.where(q_mask > 0.5, h_true, INF)
        h_pred_max = mx.where(q_mask > 0.5, h_pred, -INF)
        h_pred_min = mx.where(q_mask > 0.5, h_pred, INF)
    else:
        h_true_max = h_true
        h_true_min = h_true
        h_pred_max = h_pred
        h_pred_min = h_pred

    w_true = mx.max(h_true_max, axis=1) - mx.min(h_true_min, axis=1)
    w_pred = mx.max(h_pred_max, axis=1) - mx.min(h_pred_min, axis=1)
    width_loss = mx.mean((w_pred - w_true) ** 2)

    loss = mse + consistency_weight * consistency + width_weight * width_loss

    # --- Specific heat C(q) ---
    if y_C is not None:
        c_err = (heat_pred - y_C) ** 2
        if q_mask is not None:
            c_err = c_err * q_mask
            heat_loss = mx.sum(c_err) / mx.maximum(mx.sum(q_mask), mx.array(1.0))
        else:
            heat_loss = mx.mean(c_err)
        loss = loss + heat_weight * heat_loss

    # --- Phase transition ---
    if y_trans is not None:
        # Binary cross-entropy for has_transition
        logit = trans_pred[:, 0]
        target = y_trans[:, 0]
        bce = mx.mean(mx.maximum(logit, mx.array(0.0)) - logit * target
                       + mx.log(mx.array(1.0) + mx.exp(-mx.abs(logit))))

        # q_star regression (only for samples with transitions)
        q_star_pred = trans_pred[:, 1]
        q_star_true = y_trans[:, 1]
        q_star_mask = target  # 1.0 where there's a transition
        q_star_err = (q_star_pred - q_star_true) ** 2 * q_star_mask
        n_trans = mx.maximum(mx.sum(q_star_mask), mx.array(1.0))
        q_star_loss = mx.sum(q_star_err) / n_trans

        loss = loss + transition_weight * (bce + q_star_loss)

    return loss


# ===================================================================
# V2: Scale Weight Estimator (two-headed architecture)
# ===================================================================

class DifferentiableWeightedOLS(nn.Module):
    """Differentiable weighted ordinary least squares along scale axis.

    Given curves y(q,a), scale axis x = log2(a), and per-(q,a) weights w,
    computes weighted linear regression slopes for each q.

    All operations are differentiable so gradients flow back to the weight
    predictor.
    """

    def __call__(self, y, x, w):
        """Compute weighted OLS slopes.

        Parameters
        ----------
        y : (B, n_q, n_scales) — curve values (T, H, or D)
        x : (B, n_scales) — log2(a) scale axis
        w : (B, n_q, n_scales) — weights in [0,1]

        Returns
        -------
        slopes : (B, n_q) — fitted slopes per q
        """
        # x: (B, 1, S) for broadcasting
        x = mx.expand_dims(x, axis=1)

        S_w = mx.sum(w, axis=2)                     # (B, Q)
        S_wx = mx.sum(w * x, axis=2)                # (B, Q)
        S_wy = mx.sum(w * y, axis=2)                # (B, Q)
        S_wxx = mx.sum(w * x * x, axis=2)           # (B, Q)
        S_wxy = mx.sum(w * x * y, axis=2)           # (B, Q)

        denom = S_w * S_wxx - S_wx * S_wx
        # Avoid division by zero
        denom = mx.where(mx.abs(denom) < 1e-12, mx.array(1e-12), denom)

        slopes = (S_w * S_wxy - S_wx * S_wy) / denom
        return slopes


class ScaleWeightHead(nn.Module):
    """Predicts per-(q, scale) soft weights for weighted OLS.

    Structure:
        w_base(n_scales) : shared across q
        U(n_q, rank) @ V(rank, n_scales) : per-q adjustment (low-rank)
        w(q,a) = sigmoid(w_base + U @ V + context_bias)

    The context_bias comes from the conv features, allowing the weights
    to adapt to the specific input signal's quality at each scale.
    """

    def __init__(self, n_q, n_scales, rank=4, hidden_dim=64):
        super().__init__()
        self.n_q = n_q
        self.n_scales = n_scales

        # Base logits shared across q
        self.w_base = mx.zeros((n_scales,))

        # Low-rank per-q adjustment
        self.U = nn.Linear(n_q, rank, bias=False)  # U: (n_q,) -> (rank,)
        self.V = nn.Linear(rank, n_scales, bias=False)  # V: (rank,) -> (n_scales,)

        # Context-dependent bias from backbone features
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_scales),
        )

    def __call__(self, backbone_features):
        """Compute scale weights.

        Parameters
        ----------
        backbone_features : (B, hidden_dim) from shared backbone

        Returns
        -------
        weights : (B, n_q, n_scales) in [0, 1]
        """
        B = backbone_features.shape[0]

        # Base: (n_scales,) -> broadcast to (B, n_q, n_scales)
        base = mx.broadcast_to(self.w_base, (B, self.n_q, self.n_scales))

        # Low-rank per-q: identity-like input through U and V
        # Create q indices as one-hot-like input
        q_eye = mx.eye(self.n_q)  # (n_q, n_q)
        uv = self.V(self.U(q_eye))  # (n_q, n_scales)
        uv = mx.broadcast_to(mx.expand_dims(uv, 0), (B, self.n_q, self.n_scales))

        # Context bias from backbone (shared across q)
        ctx = self.context_proj(backbone_features)  # (B, n_scales)
        ctx = mx.expand_dims(ctx, axis=1)  # (B, 1, n_scales)

        logits = base + uv + ctx
        return mx.sigmoid(logits)


class ScaleWeightEstimator(nn.Module):
    """Two-headed model: scale weight estimator + direct spectra predictor.

    Head 1 (scale weights): Predicts per-(q, scale) weights, feeds into
    differentiable weighted OLS to compute slopes. Loss on OLS spectra
    vs analytical truth.

    Head 2 (direct): Same as SlopeEstimator — directly predicts spectra,
    C(q), and phase transitions.

    Parameters
    ----------
    n_q : int
    n_scales : int
    n_meta : int, metadata features (default 10 for v2)
    hidden_dim : int
    n_conv_channels : int
    n_input_channels : int, (default 5: T, H, D, C, n_ext)
    weight_rank : int, rank for low-rank weight adjustment
    """

    def __init__(self, n_q=47, n_scales=50, n_meta=10,
                 hidden_dim=256, n_conv_channels=32, n_input_channels=5,
                 weight_rank=4):
        super().__init__()
        self.n_q = n_q
        self.n_scales = n_scales
        self.n_input_channels = n_input_channels

        # Shared convolutional backbone
        self.conv1 = ScaleConvBlock(n_input_channels, n_conv_channels, kernel_size=7)
        self.conv2 = ScaleConvBlock(n_conv_channels, n_conv_channels, kernel_size=5)
        self.conv3 = ScaleConvBlock(n_conv_channels, n_conv_channels, kernel_size=3)

        # Direct linear slope estimate per q
        self.slope_proj = nn.Linear(n_scales, 4)

        # Shared MLP backbone
        backbone_dim = n_conv_channels * n_q + 4 * n_q + n_meta
        self.backbone = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Head 1: Scale weight estimation ---
        self.weight_head = ScaleWeightHead(
            n_q, n_scales, rank=weight_rank, hidden_dim=hidden_dim)
        self.weighted_ols = DifferentiableWeightedOLS()

        # --- Head 2: Direct spectra prediction ---
        self.spectra_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3 * n_q),
        )

        # Specific heat head: C(q)
        self.heat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_q),
        )

        # Phase transition head
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def _extract_features(self, x_curves, x_meta):
        """Shared feature extraction (same as SlopeEstimator)."""
        B = x_curves.shape[0]
        n_ch = x_curves.shape[1]

        h = self.conv1(x_curves)
        h = self.conv2(h)
        h = self.conv3(h)

        h = mx.mean(h, axis=3)  # (B, C, Q)
        h = h.reshape(B, -1)

        curves_flat = x_curves.reshape(B * n_ch * self.n_q, self.n_scales)
        slope_feat = self.slope_proj(curves_flat)
        slope_feat = slope_feat.reshape(B, n_ch, self.n_q, 4)
        slope_feat = mx.mean(slope_feat, axis=1)  # (B, Q, 4)
        slope_feat = slope_feat.reshape(B, -1)

        features = mx.concatenate([h, slope_feat, x_meta], axis=1)
        return self.backbone(features)

    def __call__(self, x_curves, x_meta, x_log2_a):
        """Forward pass.

        Parameters
        ----------
        x_curves : (B, 5, n_q, n_scales) — T, H, D, C, n_ext
        x_meta   : (B, n_meta)
        x_log2_a : (B, n_scales) — log2 of scales

        Returns
        -------
        weights    : (B, n_q, n_scales) — scale weights in [0, 1]
        spectra    : (B, 3, n_q) — direct predicted [tau, h, D]
        heat       : (B, n_q) — predicted C(q)
        transition : (B, 2) — [has_transition logit, q_star]
        """
        B = x_curves.shape[0]
        backbone_feat = self._extract_features(x_curves, x_meta)

        # Head 1: scale weights
        weights = self.weight_head(backbone_feat)

        # Head 2: direct predictions
        spectra = self.spectra_head(backbone_feat).reshape(B, 3, self.n_q)
        heat = self.heat_head(backbone_feat)
        transition = self.transition_head(backbone_feat)

        return weights, spectra, heat, transition

    def compute_wols_spectra(self, x_curves, x_log2_a, weights):
        """Compute spectra via weighted OLS on H(q,a) and D(q,a) curves.

        Uses canonical method: h(q) from H(q,a), D(q) from D(q,a),
        tau = q*h - D.

        Parameters
        ----------
        x_curves : (B, 5, n_q, n_scales)
        x_log2_a : (B, n_scales)
        weights  : (B, n_q, n_scales)

        Returns
        -------
        wols_spectra : (B, 3, n_q) — [tau, h, D] from weighted OLS
        """
        # H curves are channel 1, D curves are channel 3
        H_curves = x_curves[:, 1, :, :]  # (B, n_q, n_scales)
        D_curves = x_curves[:, 3, :, :]  # (B, n_q, n_scales)

        h_slopes = self.weighted_ols(H_curves, x_log2_a, weights)  # (B, n_q)
        D_slopes = self.weighted_ols(D_curves, x_log2_a, weights)  # (B, n_q)

        return mx.stack([
            mx.zeros_like(h_slopes),  # placeholder for tau, computed below
            h_slopes,
            D_slopes,
        ], axis=1)  # (B, 3, n_q)

    def predict(self, x_curves, x_meta, x_log2_a, q_array=None):
        """Convenience: returns numpy dicts with both heads' predictions."""
        import numpy as np
        weights, spectra, heat, transition = self(x_curves, x_meta, x_log2_a)
        wols_raw = self.compute_wols_spectra(x_curves, x_log2_a, weights)

        s = np.array(spectra)
        w = np.array(weights)
        h = np.array(heat)
        t = np.array(transition)
        wols = np.array(wols_raw)

        # Compute tau from canonical relation
        if q_array is not None:
            q = np.asarray(q_array)
            wols[:, 0, :] = q * wols[:, 1, :] - wols[:, 2, :]

        return {
            'weights': w,
            'direct': {
                'tau_q': s[:, 0, :],
                'h_q': s[:, 1, :],
                'D_q': s[:, 2, :],
            },
            'wols': {
                'tau_q': wols[:, 0, :],
                'h_q': wols[:, 1, :],
                'D_q': wols[:, 2, :],
            },
            'C_q': h,
            'has_transition': t[:, 0] > 0,
            'q_star': t[:, 1],
        }


def compute_combined_loss(model, x_curves, x_meta, x_log2_a, y_true,
                          q_array, y_C=None, y_trans=None, q_mask=None,
                          wols_weight=1.0, direct_weight=0.5,
                          consistency_weight=0.1, width_weight=0.1,
                          heat_weight=0.05, transition_weight=0.1,
                          rect_weight=0.01, smooth_weight=0.01,
                          agree_weight=0.05):
    """Combined loss for the two-headed ScaleWeightEstimator.

    L = L_wols + direct_weight * L_direct
      + consistency_weight * L_consistency
      + width_weight * L_width
      + heat_weight * L_heat + transition_weight * L_transition
      + rect_weight * L_rect + smooth_weight * L_smooth
      + agree_weight * L_agree

    Parameters
    ----------
    model : ScaleWeightEstimator
    x_curves : (B, 5, n_q, n_scales)
    x_meta : (B, n_meta)
    x_log2_a : (B, n_scales)
    y_true : (B, 3, n_q) — [tau, h, D] ground truth
    q_array : (n_q,) q values
    y_C : (B, n_q) optional
    y_trans : (B, 2) optional
    q_mask : (B, n_q) optional
    """
    weights, spectra_direct, heat_pred, trans_pred = model(
        x_curves, x_meta, x_log2_a)

    # --- Weighted OLS spectra (canonical: h from H, D from D) ---
    H_curves = x_curves[:, 1, :, :]  # (B, n_q, n_scales)
    D_curves = x_curves[:, 3, :, :]  # (B, n_q, n_scales)

    h_wols = model.weighted_ols(H_curves, x_log2_a, weights)  # (B, n_q)
    D_wols = model.weighted_ols(D_curves, x_log2_a, weights)  # (B, n_q)

    # tau = q*h - D (canonical)
    q_bc = mx.expand_dims(q_array, axis=0)  # (1, n_q)
    tau_wols = q_bc * h_wols - D_wols

    wols_spectra = mx.stack([tau_wols, h_wols, D_wols], axis=1)  # (B, 3, n_q)

    # --- Masking ---
    if q_mask is not None:
        mask_3d = mx.expand_dims(q_mask, axis=1)  # (B, 1, n_q)
        n_valid = mx.maximum(mx.sum(q_mask) * 3.0, mx.array(1.0))
        n_valid_q = mx.maximum(mx.sum(q_mask), mx.array(1.0))
    else:
        mask_3d = mx.ones_like(y_true)
        n_valid = mx.array(float(y_true.size))
        n_valid_q = mx.array(float(y_true.shape[0] * y_true.shape[2]))

    # --- L_wols: weighted OLS spectra vs truth ---
    wols_err = (wols_spectra - y_true) ** 2 * mask_3d
    L_wols = mx.sum(wols_err) / n_valid

    # --- L_direct: direct spectra vs truth ---
    direct_err = (spectra_direct - y_true) ** 2 * mask_3d
    L_direct = mx.sum(direct_err) / n_valid

    # --- L_consistency: tau = q*h - D for direct head ---
    tau_d = spectra_direct[:, 0, :]
    h_d = spectra_direct[:, 1, :]
    D_d = spectra_direct[:, 2, :]
    tau_implied = q_bc * h_d - D_d
    cons_err = (tau_d - tau_implied) ** 2
    if q_mask is not None:
        cons_err = cons_err * q_mask
    L_consistency = mx.sum(cons_err) / n_valid_q

    # --- L_width: spectrum width match (both heads) ---
    h_true = y_true[:, 1, :]
    INF = mx.array(1e6)
    if q_mask is not None:
        h_true_max = mx.where(q_mask > 0.5, h_true, -INF)
        h_true_min = mx.where(q_mask > 0.5, h_true, INF)
        h_wols_max = mx.where(q_mask > 0.5, h_wols, -INF)
        h_wols_min = mx.where(q_mask > 0.5, h_wols, INF)
    else:
        h_true_max = h_true
        h_true_min = h_true
        h_wols_max = h_wols
        h_wols_min = h_wols

    w_true = mx.max(h_true_max, axis=1) - mx.min(h_true_min, axis=1)
    w_wols = mx.max(h_wols_max, axis=1) - mx.min(h_wols_min, axis=1)
    L_width = mx.mean((w_wols - w_true) ** 2)

    # --- L_heat: specific heat ---
    L_heat = mx.array(0.0)
    if y_C is not None:
        c_err = (heat_pred - y_C) ** 2
        if q_mask is not None:
            c_err = c_err * q_mask
        L_heat = mx.sum(c_err) / n_valid_q

    # --- L_transition: phase transition detection ---
    L_transition = mx.array(0.0)
    if y_trans is not None:
        logit = trans_pred[:, 0]
        target = y_trans[:, 0]
        bce = mx.mean(mx.maximum(logit, mx.array(0.0)) - logit * target
                       + mx.log(mx.array(1.0) + mx.exp(-mx.abs(logit))))
        q_star_pred = trans_pred[:, 1]
        q_star_true = y_trans[:, 1]
        q_star_mask = target
        q_star_err = (q_star_pred - q_star_true) ** 2 * q_star_mask
        n_trans = mx.maximum(mx.sum(q_star_mask), mx.array(1.0))
        L_transition = bce + mx.sum(q_star_err) / n_trans

    # --- L_rect: rectangularity regularization (push weights to 0 or 1) ---
    L_rect = mx.mean(weights * (1.0 - weights))

    # --- L_smooth: smoothness regularization (adjacent scale weights similar) ---
    weight_diffs = weights[:, :, 1:] - weights[:, :, :-1]
    L_smooth = mx.mean(weight_diffs ** 2)

    # --- L_agree: both heads should agree ---
    agree_err = (wols_spectra - spectra_direct) ** 2 * mask_3d
    L_agree = mx.sum(agree_err) / n_valid

    # --- Total loss ---
    loss = (wols_weight * L_wols
            + direct_weight * L_direct
            + consistency_weight * L_consistency
            + width_weight * L_width
            + heat_weight * L_heat
            + transition_weight * L_transition
            + rect_weight * L_rect
            + smooth_weight * L_smooth
            + agree_weight * L_agree)

    return loss
