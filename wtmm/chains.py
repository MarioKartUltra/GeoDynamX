"""WTMM chain extraction and management across scales."""

import numpy as np


def _compile_chain_extlis():
    """Lazy-compile the numba-jitted chain_extlis function."""
    from numba import njit

    @njit(cache=True)
    def _chain_extlis_numba(prev_abs, prev_n, coarser_out, finer_in,
                            cur_abs, cur_n, finer_out, is_finest):
        """Chain finer scale (prev) to coarser scale (cur)."""
        if cur_n == 0:
            return

        nearest_start = 0

        for pi in range(prev_n):
            # Skip unchained extrema except at finest scale
            if not is_finest and finer_in[pi] == -1:
                continue

            old_dist = 9999999.0
            dist = old_dist
            dist_min = 9999999.0
            flag_nearest = False
            nearest_idx = nearest_start

            j = nearest_start
            while j < cur_n and old_dist >= dist:
                old_dist = dist
                dist = abs(cur_abs[j] - prev_abs[pi])
                if dist <= dist_min:
                    flag_nearest = True
                    dist_min = dist
                    nearest_idx = j
                j += 1

            if flag_nearest:
                # Link stealing
                existing_finer = finer_out[nearest_idx]
                if existing_finer == -1 or \
                   dist_min < abs(cur_abs[nearest_idx] - prev_abs[existing_finer]):
                    if existing_finer != -1:
                        coarser_out[existing_finer] = -1  # unlink old
                    coarser_out[pi] = nearest_idx
                    finer_out[nearest_idx] = pi
                else:
                    coarser_out[pi] = -1
            else:
                coarser_out[pi] = -1

            # Back up 2
            nearest_start = max(0, nearest_idx - 2)

    return _chain_extlis_numba


def _compile_chain_max():
    """Lazy-compile the numba-jitted chain_max function."""
    from numba import njit

    @njit(cache=True)
    def _chain_max_numba(ord_arrays, coarser_arrays, n_scales, a_min, n_voice, expo):
        """Scale-adaptive amplitude along chains starting from finest scale.
        Modifies ord_arrays in place.
        """
        n_finest = len(ord_arrays[0])

        for fi in range(n_finest):
            max_value = -1.0
            max_valueN = -1.0

            cs = 0
            ci = fi
            while cs < n_scales and ci != -1:
                new_max_valueN = abs(ord_arrays[cs][ci]) * a_min * (2.0 ** (expo * cs / n_voice))
                if new_max_valueN < max_valueN:
                    ord_arrays[cs][ci] = max_value
                else:
                    max_value = ord_arrays[cs][ci]
                    max_valueN = new_max_valueN

                ci = coarser_arrays[cs][ci]
                cs += 1

    return _chain_max_numba


_chain_extlis_jit = None
_chain_max_jit = None


def chain_extlis_numba(prev_abs, prev_n, coarser_out, finer_in,
                       cur_abs, cur_n, finer_out, is_finest):
    """Chain finer scale to coarser scale. Lazy-compiles numba JIT on first call."""
    global _chain_extlis_jit
    if _chain_extlis_jit is None:
        _chain_extlis_jit = _compile_chain_extlis()
    _chain_extlis_jit(prev_abs, prev_n, coarser_out, finer_in,
                      cur_abs, cur_n, finer_out, is_finest)


def chain_all(extrep_abs, extrep_ord, n_oct, n_voice):
    """Chain all scales. Returns (coarser_links, finer_links) — lists of int arrays."""
    n_scales = n_oct * n_voice

    # Allocate link arrays (coarser[s][i] = index into scale s+1, finer[s][i] = index into scale s-1)
    coarser_links = []
    finer_links = []
    for s in range(n_scales):
        n = len(extrep_abs[s])
        coarser_links.append(np.full(n, -1, dtype=np.int64))
        finer_links.append(np.full(n, -1, dtype=np.int64))

    # Chain from finest to coarsest
    for s in range(1, n_scales):
        is_finest = (s == 1)
        prev_n = len(extrep_abs[s - 1])
        cur_n = len(extrep_abs[s])
        if prev_n == 0 or cur_n == 0:
            continue
        chain_extlis_numba(
            extrep_abs[s - 1], prev_n, coarser_links[s - 1], finer_links[s - 1],
            extrep_abs[s], cur_n, finer_links[s], is_finest)

    return coarser_links, finer_links


def save_chain_state(extrep_abs, extrep_ord, coarser_links):
    """Save a deep copy of chain state before deletion."""
    return ([a.copy() for a in extrep_abs],
            [o.copy() for o in extrep_ord],
            [c.copy() for c in coarser_links])


def chain_delete_all(extrep_abs, extrep_ord, extrep_idx,
                     coarser_links, finer_links, n_oct, n_voice):
    """Delete unchained and sign-changing extrema. Modifies arrays in place.
    Returns number of deletion triggers.

    Matches LastWave's ChainDelete() + RemoveDeleteChain(): for each scale > 0,
    if an extremum has no finer link OR a sign change with its finer neighbor,
    disconnect the finer link and delete the ENTIRE chain from that point
    through all coarser scales.
    """
    n_scales = n_oct * n_voice
    nb_deleted = 0

    # Pass 1: Mark all extrema to delete.
    # to_delete[s] = set of indices at scale s that should be removed.
    to_delete = [set() for _ in range(n_scales)]

    for s in range(1, n_scales):
        for i in range(len(extrep_abs[s])):
            if i in to_delete[s]:
                continue
            fi = finer_links[s][i]
            if fi == -1 or extrep_ord[s][i] * extrep_ord[s - 1][fi] < 0:
                # Disconnect finer link
                if fi != -1:
                    coarser_links[s - 1][fi] = -1
                    finer_links[s][i] = -1

                # Mark this extremum and entire coarser chain for deletion
                cs, ci = s, i
                while cs < n_scales and ci != -1:
                    to_delete[cs].add(ci)
                    next_ci = coarser_links[cs][ci]
                    # Disconnect links at this node
                    if next_ci != -1 and cs + 1 < n_scales:
                        finer_links[cs + 1][next_ci] = -1
                        coarser_links[cs][ci] = -1
                    ci = next_ci
                    cs += 1

                nb_deleted += 1

    # Pass 2: Remove marked extrema and remap indices, finest to coarsest.
    for s in range(1, n_scales):
        if not to_delete[s]:
            continue

        n = len(extrep_abs[s])
        keep = np.ones(n, dtype=np.bool_)
        for idx in to_delete[s]:
            keep[idx] = False

        # Build index remapping: new_indices[old] = new position
        new_indices = np.cumsum(keep) - 1

        # Update finer links in scale s+1 that point into scale s
        if s + 1 < n_scales:
            for j in range(len(finer_links[s + 1])):
                old_fi = finer_links[s + 1][j]
                if old_fi != -1:
                    if keep[old_fi]:
                        finer_links[s + 1][j] = new_indices[old_fi]
                    else:
                        finer_links[s + 1][j] = -1

        # Update coarser links in scale s-1 that point into scale s
        for j in range(len(coarser_links[s - 1])):
            old_ci = coarser_links[s - 1][j]
            if old_ci != -1:
                if keep[old_ci]:
                    coarser_links[s - 1][j] = new_indices[old_ci]
                else:
                    coarser_links[s - 1][j] = -1

        # Filter arrays for this scale
        extrep_abs[s] = extrep_abs[s][keep]
        extrep_ord[s] = extrep_ord[s][keep]
        extrep_idx[s] = extrep_idx[s][keep]
        coarser_links[s] = coarser_links[s][keep]
        finer_links[s] = finer_links[s][keep]

    return nb_deleted


def chain_max_numba(ord_arrays, coarser_arrays, n_scales, a_min, n_voice, expo):
    """Scale-adaptive amplitude along chains. Lazy-compiles numba JIT on first call."""
    global _chain_max_jit
    if _chain_max_jit is None:
        _chain_max_jit = _compile_chain_max()
    _chain_max_jit(ord_arrays, coarser_arrays, n_scales, a_min, n_voice, expo)


def chain_max_wrapper(extrep_ord, coarser_links, n_oct, n_voice, a_min, expo=1.0):
    """Wrapper to call chain_max_numba with typed lists."""
    from numba.typed import List as NumbaList
    n_scales = n_oct * n_voice
    ord_list = NumbaList()
    coarser_list = NumbaList()
    for s in range(n_scales):
        ord_list.append(extrep_ord[s])
        coarser_list.append(coarser_links[s])
    chain_max_numba(ord_list, coarser_list, n_scales, a_min, n_voice, expo)


def trace_chains(extrep_abs, extrep_ord, coarser_links, scales, n_scales):
    """Trace all chains from finest scale.

    Returns list of (positions, log2scales, sign, ordinates) tuples.
    For backwards compatibility, tuples can be unpacked as 3-element
    (positions, log2scales, sign) — the 4th element is extra.
    """
    chains = []
    for fi in range(len(extrep_abs[0])):
        positions = []
        scale_vals = []
        ordinates = []
        cs = 0
        ci = fi
        while cs < n_scales and ci != -1:
            positions.append(extrep_abs[cs][ci])
            scale_vals.append(np.log2(scales[cs]))
            ordinates.append(extrep_ord[cs][ci])
            ci = coarser_links[cs][ci]
            cs += 1
        if len(positions) > 1:
            sign = 1 if extrep_ord[0][fi] > 0 else -1
            chains.append((positions, scale_vals, sign, ordinates))
    return chains
