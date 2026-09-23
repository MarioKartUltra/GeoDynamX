# wtmm_ebsd

EBSD-specific extensions to the `wtmm` multifractal toolkit. Sister package to `wtmm/` and `xsmurf_wrapper/`.

## Status

Bootstrap phase (Phase 0b of the refactor). Module skeletons coming online incrementally per the roadmap in `../PERGRAIN_PANTLEON_REFACTOR_PLAN.md`.

## Package layout (planned)

```
wtmm_ebsd/
├── orientation.py     # quat ops, Karcher mean, frame rotations
├── symmetry.py        # Laue groups, FZ thresholds, Dauphine
├── grains.py          # flood-fill, smart fill, GB masks
├── demean.py          # per-grain log orientation (crystal + sample frame)
├── nye.py             # Pantleon Eqs. 11/13, GB-aware FD (Numba)
├── kam.py             # orix KAM + Lebyodkin Mode A
├── cpo.py             # c-axis attractor, directional anisotropy
├── papeschi.py        # quartz Dauphine, MAD, MOCC, MOSC
├── twtmm.py           # alpha-Jacobian TWTMM driver
├── shuffles.py        # Kantelhardt up/downstream shuffles
├── chain_filters.py   # modulus & Hölder thresholds
├── plot_overlays.py   # GB overlay widgets, palettes
├── slip_canon.py      # phase-name canonicalization (may merge into wtmm.slip_systems)
└── tests/             # snapshot regression tests
```

## Install

```bash
cd <path-to>/wavelet
pip install -e ./wtmm_ebsd
```

## Run regression tests

```bash
cd <path-to>/wavelet/wtmm_ebsd
pytest tests/ -v
```

Tests load the v1 snapshot from `../_backups/snapshot_v1_dataset20.npz`. If that file doesn't exist, snapshot-dependent tests skip with a clear message.

## Tolerance policy

See `../PERGRAIN_PANTLEON_REFACTOR_PLAN.md` § "Risk / regression strategy" for the per-array tolerance table. Briefly:

- Float32 arrays + extraction-only (no math change): **bit-exact** (`np.array_equal`).
- Float32 arrays + vectorization or Numba parallel: `rtol=1e-6`.
- Float64 arrays + extraction: bit-exact.
- Float64 arrays + vectorization/Numba: `rtol=1e-10` (most arrays) or `rtol=1e-8` (reduction-heavy).
- LP per-system `rho`: NOT tested (basis is unstable). Aggregate `gnd_per_family` is.
