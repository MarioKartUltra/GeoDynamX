# DynamiX

## GeoDynamix_Beta — install (Apple Silicon macOS or Windows)

Use a **fresh environment** — never the one DynamiX itself is installed in (both import as
`dynamix`):

```bash
conda create -n geodynamix python=3.12
conda activate geodynamix
pip install -e ".[app,dev]"
python -m dynamix.shell.app
```

On Apple Silicon this installs **mlx** (GPU FFTs); on Windows and everywhere else every FFT runs on
**FFTW3** (pyfftw) — no numpy FFT, no mlx. Settings ▸ *Compute engine* and *FFT precision* pick the
engine and 32/64-bit (64-bit always uses FFTW3); restart to apply. No private packages are needed:
`wtmm` / `wtmm_ebsd` are vendored under `src/dynamix/_vendor/` (EBSD reading is not part of the
beta). Optional extras: `catalog` (data-portal tooling, pyyaml), `wtmm-extras` (the vendored
wtmm's NN/dataset helpers).


A multiscale structure instrument for structural geology. Load a raster of any provenance — DEM,
bathymetry, EBSD map, thin section, outcrop photo, satellite imagery — build a chain of operations
on it, and watch the fabric respond as you turn the knobs. The map, the rose diagram, the stereonet
and the D(h) spectrum are all live views of the same chain.

Analysis runs in each dataset's native domain; results display on a WGS84 globe.

**Status: early, but it runs.** The analysis core is extracted, the device/document model is built,
and there is a working demo of the core gesture.

## Run the demo

```bash
scripts/demo.sh                       # the in-repo EBSD fixture (64x64 at 70 µm/px)
scripts/demo.sh path/to/field.npz     # any RasterField .npz
```

Move the **Scale** slider and watch the fabric coarsen; set **Strike** to 040 with **± width** 15
and watch it select one orientation family. The readout reports, honestly, whether a redraw touched
the transform or only the filters.

**What it demonstrates.** The WTMM transform runs once (~1.2 s on the fixture) and is cached. Every
control after it is a filter, so a redraw is a lookup:

| state | extrema shown | redraw | cache |
|---|---|---|---|
| scale 0, all | 1127 | 0.1 ms | hit |
| scale 4, all | 619 | 0.1 ms | hit |
| scale 8, all | 206 | 0.1 ms | hit |
| scale 11, all | 20 | 0.1 ms | hit |
| scale 2, wedge 040 ± 15 | 80 | 0.1 ms | hit |
| scale 2, wedge 130 ± 15 | 189 | 0.1 ms | hit |

Extrema falling off 1127 → 20 as scale coarsens is the fabric resolving fewer features; the two
wedges selecting different counts is anisotropy. `scripts/render_demo_states.py` renders each
state.

Every control is **generated from its device's `Param` declaration** — no widget is hand-written
per device. Sliders span each param's *soft* range while the box accepts its full *hard* range, so
typing a value outside the slider widens the view rather than erroring.

Regenerate the renders headlessly (no display needed) with:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/render_demo_states.py /tmp/dxdemo
```

## Device reference

[`docs/reference/devices_reference.pdf`](docs/reference/devices_reference.pdf) documents every
tool and every knob, the $L^1$ wavelet normalization, and pictures of the wavelet families
(q-Gaussian and q-Mexican hat, fractional B-splines, fractional Gaussians, the WTMM smoothing
function and analyzing wavelet). The pictures are drawn from the app's own kernel code:
`python docs/reference/make_figures.py`, then `latexmk -pdf devices_reference.tex` in that folder.

## Install options

The analysis core is headless and needs only numpy and FFTW3 (plus mlx on Apple Silicon):

```bash
pip install -e .           # analysis core
pip install -e ".[app]"    # + the desktop app: Qt, pyqtgraph, pyvista, rasterio, numba/scipy
```

`[fast]`, `[io]`, `[gui]` and `[viz]` install the pieces of `[app]` separately.

### The wavelet toolkit

The 1-D WTMM multifractal toolkit (`wtmm`) and its 2-D extensions (`wtmm_ebsd`, without EBSD
reading) are vendored under `src/dynamix/_vendor/`, so there is nothing extra to install. They are
imported lazily, so the analysis core imports and passes its tests without the optional extras.

## Tests

```bash
python -m pytest
```

`tests/test_no_silent_drift.py` hashes each module copied from EQSelect against its original, so
divergence becomes a decision someone makes on purpose. It skips when the EQSelect checkout is
absent.

## License and citation

Copyright (C) 2026 Abraham Joseph Okayli Masaryk. GeoDynamix is free software under the GNU
General Public License (see [`LICENSE`](LICENSE)): the author's code is GPL-2.0-or-later, and the
modules translated from **xsmurf** (Decoster, Kestener, Roux, Arneodo; GPL-2.0) are GPL-2.0-only.
The 1-D WTMM code is based on **LastWave** (Bacry, Mallat; GPL-2.0-or-later). See
[`NOTICE`](NOTICE) for the derived code and the principal scientific references, and
[`CITATION.cff`](CITATION.cff) for how to cite this software.

## Relationship to EQSelect

The analysis core is a **copy** from EQSelect, not a dependency. That application is in active use
and is never modified by work here. The cost — two copies of a 2,323-line WTMM backend will drift —
is accepted, and the drift test makes it visible.
