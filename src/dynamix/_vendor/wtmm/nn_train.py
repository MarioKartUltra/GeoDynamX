"""Training loop, evaluation, and metrics for the NN slope estimator."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .nn_model import SlopeEstimator, compute_loss_with_q
from .nn_model import ScaleWeightEstimator, compute_combined_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spectrum_width(h_q, mask=None):
    """Compute spectrum width: max(h) - min(h) over valid q.

    Parameters
    ----------
    h_q : array of shape (..., n_q) — Holder exponents
    mask : array of shape (..., n_q) or None — 1 where valid

    Returns
    -------
    width : array of shape (...)
    """
    h = np.asarray(h_q)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    else:
        mask = np.isfinite(h)

    if h.ndim == 1:
        hf = h[mask]
        return hf.max() - hf.min() if len(hf) > 0 else 0.0
    # Batch
    widths = np.zeros(h.shape[:-1])
    for idx in np.ndindex(h.shape[:-1]):
        hf = h[idx][mask[idx]]
        widths[idx] = (hf.max() - hf.min()) if len(hf) > 0 else 0.0
    return widths


def compute_metrics(y_pred, y_true, q_array=None, q_mask=None):
    """Compute evaluation metrics comparing predicted vs true spectra.

    Parameters
    ----------
    y_pred : (N, 3, n_q) — predicted [tau, h, D]
    y_true : (N, 3, n_q) — ground truth [tau, h, D]
    q_array : (n_q,) optional
    q_mask : (N, n_q) optional — 1.0 where valid

    Returns
    -------
    metrics : dict
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if q_mask is not None:
        q_mask = np.asarray(q_mask)
        mask_3d = q_mask[:, np.newaxis, :]  # (N, 1, n_q)
    else:
        mask_3d = np.ones_like(y_true)
        q_mask = np.ones(y_true.shape[0:1] + y_true.shape[2:3])

    metrics = {}

    for i, name in enumerate(['tau', 'h', 'D']):
        diff = y_pred[:, i, :] - y_true[:, i, :]
        m = q_mask
        masked_sq = (diff ** 2) * m
        masked_abs = np.abs(diff) * m
        n_valid = m.sum()
        metrics[f'mse_{name}'] = float(masked_sq.sum() / max(n_valid, 1))
        metrics[f'mae_{name}'] = float(masked_abs.sum() / max(n_valid, 1))

    all_diff = (y_pred - y_true) ** 2 * mask_3d
    metrics['mse_total'] = float(all_diff.sum() / max(mask_3d.sum(), 1))

    # Spectrum width (using mask)
    w_true = spectrum_width(y_true[:, 1, :], q_mask > 0.5)
    w_pred = spectrum_width(y_pred[:, 1, :], q_mask > 0.5)
    metrics['width_true'] = w_true
    metrics['width_pred'] = w_pred
    metrics['width_mae'] = float(np.mean(np.abs(w_pred - w_true)))
    metrics['width_bias'] = float(np.mean(w_pred - w_true))

    return metrics


def per_signal_metrics(dataset, y_pred, q_mask=None):
    """Group metrics by signal type."""
    from collections import defaultdict

    groups = defaultdict(lambda: {'pred': [], 'true': [], 'mask': []})
    for i, sample in enumerate(dataset):
        name = sample['info']['signal_name']
        groups[name]['pred'].append(y_pred[i])
        groups[name]['true'].append(
            np.stack([sample['labels']['tau_q'],
                      sample['labels']['h_q'],
                      sample['labels']['D_q']]))
        if q_mask is not None:
            groups[name]['mask'].append(q_mask[i])

    result = {}
    for name, data in groups.items():
        pred = np.array(data['pred'])
        true = np.array(data['true'])
        m = np.array(data['mask']) if len(data['mask']) > 0 else None
        result[name] = compute_metrics(pred, true, q_mask=m)

    return result


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_val_split(dataset, val_fraction=0.2, seed=42):
    """Split dataset stratified by signal type."""
    from collections import defaultdict

    rng = np.random.default_rng(seed)

    groups = defaultdict(list)
    for i, sample in enumerate(dataset):
        groups[sample['info']['signal_name']].append(i)

    train_idx = []
    val_idx = []

    for name, indices in groups.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx.extend(indices[:n_val].tolist())
        train_idx.extend(indices[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(dataset, n_q, n_scales, q_array,
          epochs=200, lr=1e-3, batch_size=32,
          val_fraction=0.2, patience=20,
          consistency_weight=0.1, width_weight=0.1,
          heat_weight=0.05, transition_weight=0.1,
          hidden_dim=256, n_conv_channels=32,
          seed=42, verbose=True):
    """Train the SlopeEstimator on a dataset.

    Returns
    -------
    model : SlopeEstimator (best weights)
    history : dict with 'train_loss', 'val_loss', 'val_metrics' per epoch
    """
    from .datasets import dataset_to_arrays

    # Convert to arrays (4 channels now)
    X_curves, X_meta, Y, Y_C, Y_trans, Q_mask, Y_ols = dataset_to_arrays(
        dataset, n_q, n_scales)

    # Split — keep as NumPy, convert per-batch to avoid MLX indexing issues
    train_idx, val_idx = train_val_split(dataset, val_fraction, seed)

    X_curves_train = X_curves[train_idx]
    X_meta_train = X_meta[train_idx]
    Y_train = Y[train_idx]
    Y_C_train = Y_C[train_idx]
    Y_trans_train = Y_trans[train_idx]
    Q_mask_train = Q_mask[train_idx]

    X_curves_val = X_curves[val_idx]
    X_meta_val = X_meta[val_idx]
    Y_val = Y[val_idx]
    Y_C_val = Y_C[val_idx]
    Y_trans_val = Y_trans[val_idx]
    Q_mask_val = Q_mask[val_idx]

    q_mx = mx.array(q_array.astype(np.float32))

    N_train = len(train_idx)
    N_val = len(val_idx)

    # Model
    model = SlopeEstimator(
        n_q=n_q, n_scales=n_scales, n_input_channels=4,
        hidden_dim=hidden_dim, n_conv_channels=n_conv_channels)

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x_c, x_m, y, y_c, y_t, q_m):
        return compute_loss_with_q(
            model, x_c, x_m, y, q_mx,
            y_C=y_c, y_trans=y_t, q_mask=q_m,
            consistency_weight=consistency_weight,
            width_weight=width_weight,
            heat_weight=heat_weight,
            transition_weight=transition_weight)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
    }

    from mlx.utils import tree_map

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        perm = np.random.default_rng(seed + epoch).permutation(N_train)
        epoch_loss = 0.0
        n_batches = 0

        model.train()

        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            idx = perm[start:end]

            loss, grads = loss_and_grad(
                model,
                mx.array(X_curves_train[idx]), mx.array(X_meta_train[idx]),
                mx.array(Y_train[idx]), mx.array(Y_C_train[idx]),
                mx.array(Y_trans_train[idx]), mx.array(Q_mask_train[idx]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_preds = []

        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            xc_b = mx.array(X_curves_val[start:end])
            xm_b = mx.array(X_meta_val[start:end])
            y_b = mx.array(Y_val[start:end])
            yc_b = mx.array(Y_C_val[start:end])
            yt_b = mx.array(Y_trans_val[start:end])
            qm_b = mx.array(Q_mask_val[start:end])

            loss = loss_fn(model, xc_b, xm_b, y_b, yc_b, yt_b, qm_b)
            val_loss += loss.item()
            val_batches += 1

            spectra, _, _ = model(xc_b, xm_b)
            val_preds.append(np.array(spectra))

        avg_val_loss = val_loss / max(val_batches, 1)

        val_pred_all = np.concatenate(val_preds, axis=0)
        val_true_all = Y_val
        val_mask_all = Q_mask_val
        val_met = compute_metrics(val_pred_all, val_true_all,
                                  q_array, q_mask=val_mask_all)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_met)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f'Epoch {epoch:3d} | train={avg_train_loss:.6f} '
                  f'val={avg_val_loss:.6f} | '
                  f'MAE h={val_met["mae_h"]:.4f} '
                  f'D={val_met["mae_D"]:.4f} '
                  f'width={val_met["width_mae"]:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Deep copy weights as a flat list of (name, array) pairs
            best_weights = tree_map(lambda x: mx.array(x), model.parameters())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

    if best_weights is not None:
        model.update(best_weights)

    if verbose:
        print(f'Best val loss: {best_val_loss:.6f}')

    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataset, n_q, n_scales, q_array, batch_size=64):
    """Run model on full dataset and compute metrics + per-signal breakdown."""
    from .datasets import dataset_to_arrays

    X_curves, X_meta, Y, Y_C, Y_trans, Q_mask, Y_ols = dataset_to_arrays(
        dataset, n_q, n_scales)

    model.eval()
    spec_preds = []
    heat_preds = []
    trans_preds = []
    N = len(dataset)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_c = mx.array(X_curves[start:end])
        x_m = mx.array(X_meta[start:end])
        spectra, heat, trans = model(x_c, x_m)
        spec_preds.append(np.array(spectra))
        heat_preds.append(np.array(heat))
        trans_preds.append(np.array(trans))

    predictions = np.concatenate(spec_preds, axis=0)
    heat_predictions = np.concatenate(heat_preds, axis=0)
    trans_predictions = np.concatenate(trans_preds, axis=0)

    metrics = compute_metrics(predictions, Y, q_array, q_mask=Q_mask)
    per_sig = per_signal_metrics(dataset, predictions, q_mask=Q_mask)

    ols_metrics = None
    if Y_ols is not None:
        ols_metrics = compute_metrics(Y_ols, Y, q_array, q_mask=Q_mask)

    # Phase transition accuracy
    trans_true = Y_trans[:, 0] > 0.5
    trans_pred_bool = trans_predictions[:, 0] > 0
    trans_acc = float(np.mean(trans_true == trans_pred_bool))
    metrics['transition_accuracy'] = trans_acc

    # q_star error (only where transitions exist)
    if trans_true.any():
        q_star_err = np.abs(
            trans_predictions[trans_true, 1] - Y_trans[trans_true, 1])
        metrics['q_star_mae'] = float(np.mean(q_star_err))

    return {
        'predictions': predictions,
        'heat_predictions': heat_predictions,
        'trans_predictions': trans_predictions,
        'metrics': metrics,
        'per_signal': per_sig,
        'ols_metrics': ols_metrics,
        'Q_mask': Q_mask,
    }


def compare_nn_vs_ols(dataset, predictions, q_array):
    """Compare NN predictions vs OLS slopes vs analytical truth."""
    N = len(dataset)
    n_q = len(q_array)

    true = np.zeros((N, 3, n_q))
    ols = np.zeros((N, 3, n_q))
    q_masks = np.ones((N, n_q))
    has_ols = False
    names = []

    for i, sample in enumerate(dataset):
        names.append(sample['info']['signal_name'])
        for j, key in enumerate(['tau_q', 'h_q', 'D_q']):
            arr = np.array(sample['labels'][key])
            true[i, j, :min(len(arr), n_q)] = arr[:n_q]

        if 'q_valid_mask' in sample['labels']:
            m = np.array(sample['labels']['q_valid_mask'], dtype=float)
            q_masks[i, :min(len(m), n_q)] = m[:n_q]

        if sample['ols'] is not None:
            has_ols = True
            for j, key in enumerate(['tau_q', 'h_q', 'D_q']):
                arr = np.array(sample['ols'][key])
                ols[i, j, :min(len(arr), n_q)] = arr[:n_q]

    result = {
        'nn': predictions,
        'ols': ols if has_ols else None,
        'true': true,
        'q': q_array,
        'q_masks': q_masks,
        'signal_names': names,
        'nn_width': spectrum_width(predictions[:, 1, :], q_masks > 0.5),
        'true_width': spectrum_width(true[:, 1, :], q_masks > 0.5),
    }

    if has_ols:
        result['ols_width'] = spectrum_width(ols[:, 1, :], q_masks > 0.5)
    else:
        result['ols_width'] = None

    return result


def save_model(model, path):
    """Save model weights to a file."""
    model.save_weights(path)


def load_model(path, n_q=47, n_scales=50, **kwargs):
    """Load model weights from a file."""
    model = SlopeEstimator(n_q=n_q, n_scales=n_scales, **kwargs)
    model.load_weights(path)
    return model


# ---------------------------------------------------------------------------
# V2: Training with ScaleWeightEstimator
# ---------------------------------------------------------------------------

def train_v2(dataset, n_q, n_scales, q_array,
             epochs=200, lr=1e-3, batch_size=32,
             val_fraction=0.2, patience=20,
             wols_weight=1.0, direct_weight=0.5,
             consistency_weight=0.1, width_weight=0.1,
             heat_weight=0.05, transition_weight=0.1,
             rect_weight=0.01, smooth_weight=0.01,
             agree_weight=0.05,
             hidden_dim=256, n_conv_channels=32,
             weight_rank=4,
             seed=42, verbose=True):
    """Train the ScaleWeightEstimator on a dataset.

    Returns
    -------
    model : ScaleWeightEstimator (best weights)
    history : dict with 'train_loss', 'val_loss', 'val_metrics' per epoch
    """
    from .datasets import dataset_to_arrays_v2

    X_curves, X_meta, Y, Y_C, Y_trans, Q_mask, X_log2_a, Y_ols = \
        dataset_to_arrays_v2(dataset, n_q, n_scales)

    train_idx, val_idx = train_val_split(dataset, val_fraction, seed)

    X_curves_train = X_curves[train_idx]
    X_meta_train = X_meta[train_idx]
    Y_train = Y[train_idx]
    Y_C_train = Y_C[train_idx]
    Y_trans_train = Y_trans[train_idx]
    Q_mask_train = Q_mask[train_idx]
    X_log2_a_train = X_log2_a[train_idx]

    X_curves_val = X_curves[val_idx]
    X_meta_val = X_meta[val_idx]
    Y_val = Y[val_idx]
    Y_C_val = Y_C[val_idx]
    Y_trans_val = Y_trans[val_idx]
    Q_mask_val = Q_mask[val_idx]
    X_log2_a_val = X_log2_a[val_idx]

    q_mx = mx.array(q_array.astype(np.float32))
    N_train = len(train_idx)
    N_val = len(val_idx)

    model = ScaleWeightEstimator(
        n_q=n_q, n_scales=n_scales, n_meta=10,
        n_input_channels=5,
        hidden_dim=hidden_dim, n_conv_channels=n_conv_channels,
        weight_rank=weight_rank)

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x_c, x_m, x_la, y, y_c, y_t, q_m):
        return compute_combined_loss(
            model, x_c, x_m, x_la, y, q_mx,
            y_C=y_c, y_trans=y_t, q_mask=q_m,
            wols_weight=wols_weight, direct_weight=direct_weight,
            consistency_weight=consistency_weight,
            width_weight=width_weight,
            heat_weight=heat_weight,
            transition_weight=transition_weight,
            rect_weight=rect_weight, smooth_weight=smooth_weight,
            agree_weight=agree_weight)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    from mlx.utils import tree_map

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        perm = np.random.default_rng(seed + epoch).permutation(N_train)
        epoch_loss = 0.0
        n_batches = 0

        model.train()

        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            idx = perm[start:end]

            loss, grads = loss_and_grad(
                model,
                mx.array(X_curves_train[idx]),
                mx.array(X_meta_train[idx]),
                mx.array(X_log2_a_train[idx]),
                mx.array(Y_train[idx]),
                mx.array(Y_C_train[idx]),
                mx.array(Y_trans_train[idx]),
                mx.array(Q_mask_train[idx]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_wols_preds = []
        val_direct_preds = []

        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            xc_b = mx.array(X_curves_val[start:end])
            xm_b = mx.array(X_meta_val[start:end])
            xla_b = mx.array(X_log2_a_val[start:end])
            y_b = mx.array(Y_val[start:end])
            yc_b = mx.array(Y_C_val[start:end])
            yt_b = mx.array(Y_trans_val[start:end])
            qm_b = mx.array(Q_mask_val[start:end])

            loss = loss_fn(model, xc_b, xm_b, xla_b, y_b, yc_b, yt_b, qm_b)
            val_loss += loss.item()
            val_batches += 1

            weights, spectra_d, _, _ = model(xc_b, xm_b, xla_b)
            val_direct_preds.append(np.array(spectra_d))

            # WOLS spectra
            wols_raw = model.compute_wols_spectra(xc_b, xla_b, weights)
            wols_np = np.array(wols_raw)
            # Compute tau = q*h - D
            q_np = q_array.astype(np.float32)
            wols_np[:, 0, :] = q_np * wols_np[:, 1, :] - wols_np[:, 2, :]
            val_wols_preds.append(wols_np)

        avg_val_loss = val_loss / max(val_batches, 1)

        val_direct_all = np.concatenate(val_direct_preds, axis=0)
        val_wols_all = np.concatenate(val_wols_preds, axis=0)

        val_met = compute_metrics(val_wols_all, Y_val, q_array,
                                  q_mask=Q_mask_val)
        val_met_direct = compute_metrics(val_direct_all, Y_val, q_array,
                                         q_mask=Q_mask_val)
        val_met['direct_mse_h'] = val_met_direct['mse_h']
        val_met['direct_mae_h'] = val_met_direct['mae_h']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_met)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f'Epoch {epoch:3d} | train={avg_train_loss:.6f} '
                  f'val={avg_val_loss:.6f} | '
                  f'WOLS MAE h={val_met["mae_h"]:.4f} '
                  f'D={val_met["mae_D"]:.4f} '
                  f'width={val_met["width_mae"]:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = tree_map(lambda x: mx.array(x), model.parameters())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

    if best_weights is not None:
        model.update(best_weights)

    if verbose:
        print(f'Best val loss: {best_val_loss:.6f}')

    return model, history


def evaluate_v2(model, dataset, n_q, n_scales, q_array, batch_size=64):
    """Run ScaleWeightEstimator on full dataset and compute metrics."""
    from .datasets import dataset_to_arrays_v2

    X_curves, X_meta, Y, Y_C, Y_trans, Q_mask, X_log2_a, Y_ols = \
        dataset_to_arrays_v2(dataset, n_q, n_scales)

    model.eval()
    wols_preds = []
    direct_preds = []
    weight_preds = []
    heat_preds = []
    trans_preds = []
    N = len(dataset)

    q_np = q_array.astype(np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_c = mx.array(X_curves[start:end])
        x_m = mx.array(X_meta[start:end])
        x_la = mx.array(X_log2_a[start:end])

        weights, spectra_d, heat, trans = model(x_c, x_m, x_la)
        wols_raw = model.compute_wols_spectra(x_c, x_la, weights)

        wols_np = np.array(wols_raw)
        wols_np[:, 0, :] = q_np * wols_np[:, 1, :] - wols_np[:, 2, :]

        wols_preds.append(wols_np)
        direct_preds.append(np.array(spectra_d))
        weight_preds.append(np.array(weights))
        heat_preds.append(np.array(heat))
        trans_preds.append(np.array(trans))

    wols_all = np.concatenate(wols_preds, axis=0)
    direct_all = np.concatenate(direct_preds, axis=0)
    weights_all = np.concatenate(weight_preds, axis=0)

    wols_metrics = compute_metrics(wols_all, Y, q_array, q_mask=Q_mask)
    direct_metrics = compute_metrics(direct_all, Y, q_array, q_mask=Q_mask)

    ols_metrics = None
    if Y_ols is not None:
        ols_metrics = compute_metrics(Y_ols, Y, q_array, q_mask=Q_mask)

    # Phase transition accuracy
    trans_all = np.concatenate(trans_preds, axis=0)
    trans_true = Y_trans[:, 0] > 0.5
    trans_pred_bool = trans_all[:, 0] > 0
    wols_metrics['transition_accuracy'] = float(
        np.mean(trans_true == trans_pred_bool))

    return {
        'wols_predictions': wols_all,
        'direct_predictions': direct_all,
        'weights': weights_all,
        'heat_predictions': np.concatenate(heat_preds, axis=0),
        'trans_predictions': trans_all,
        'wols_metrics': wols_metrics,
        'direct_metrics': direct_metrics,
        'ols_metrics': ols_metrics,
        'Q_mask': Q_mask,
        'X_log2_a': X_log2_a,
    }


# ---------------------------------------------------------------------------
# Weight visualization
# ---------------------------------------------------------------------------

def plot_scale_weights(weights, log2_a, q_array, signal_name='',
                       q_indices=None, fig=None, axes=None):
    """Visualize learned scale weights for a single sample.

    Parameters
    ----------
    weights : (n_q, n_scales) — scale weights in [0, 1]
    log2_a : (n_scales,) — log2 of scales
    q_array : (n_q,) — q values
    signal_name : str
    q_indices : list of int, optional — which q indices to plot as lines
    fig, axes : optional

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(weights, aspect='auto', origin='lower',
                   extent=[log2_a[0], log2_a[-1], q_array[0], q_array[-1]],
                   vmin=0, vmax=1, cmap='viridis')
    fig.colorbar(im, ax=ax, label='Weight')
    ax.set_xlabel('log2(a)')
    ax.set_ylabel('q')
    ax.set_title(f'Scale weights — {signal_name}' if signal_name
                 else 'Scale weights w(q,a)')

    # Line plots for selected q values
    ax = axes[1]
    if q_indices is None:
        # Default: q ≈ -2, 0, 1, 2, 3
        targets = [-2, 0, 1, 2, 3]
        q_indices = [np.argmin(np.abs(q_array - t)) for t in targets]
        q_indices = sorted(set(q_indices))

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(q_indices)))
    for ci, qi in enumerate(q_indices):
        ax.plot(log2_a, weights[qi], color=colors[ci],
                label=f'q={q_array[qi]:.1f}', linewidth=1.5)
    ax.set_xlabel('log2(a)')
    ax.set_ylabel('Weight')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title('Weight profiles per q')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_weight_summary(eval_result, dataset, q_array, n_examples=4):
    """Plot weight summaries for a few representative signals.

    Parameters
    ----------
    eval_result : dict from evaluate_v2
    dataset : list of sample dicts
    q_array : (n_q,)
    n_examples : int

    Returns
    -------
    fig
    """
    import matplotlib.pyplot as plt

    weights = eval_result['weights']
    log2_a = eval_result['X_log2_a']

    # Pick diverse signals
    seen_names = set()
    indices = []
    for i, sample in enumerate(dataset):
        name = sample['info']['signal_name']
        if name not in seen_names:
            seen_names.add(name)
            indices.append(i)
        if len(indices) >= n_examples:
            break

    fig, all_axes = plt.subplots(n_examples, 2, figsize=(14, 4 * n_examples))
    if n_examples == 1:
        all_axes = [all_axes]

    for row, idx in enumerate(indices):
        name = dataset[idx]['info']['signal_name']
        plot_scale_weights(weights[idx], log2_a[idx], q_array,
                           signal_name=name,
                           fig=fig, axes=all_axes[row])

    plt.tight_layout()
    return fig


def save_model_v2(model, path):
    """Save ScaleWeightEstimator weights."""
    model.save_weights(path)


def load_model_v2(path, n_q=47, n_scales=50, **kwargs):
    """Load ScaleWeightEstimator weights."""
    model = ScaleWeightEstimator(n_q=n_q, n_scales=n_scales, **kwargs)
    model.load_weights(path)
    return model
