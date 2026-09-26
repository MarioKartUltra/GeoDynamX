# GeoDynamix (beta)

GeoDynamix is a desktop application for the multiscale analysis of geoscience rasters: digital
elevation models, bathymetry, potential-field grids, and satellite or outcrop imagery. It measures
how structure changes with scale, using wavelet-based multifractal methods, edge detection and
matrix decompositions, and shows the results on a globe in their true geographic position.

Each dataset is analysed on its own native grid. Only the results are projected, because
resampling a raster before a scale-sensitive analysis changes the statistics being measured.

## What it does

- **Multifractal analysis (canonical).** The 2-D wavelet transform modulus maxima method (WTMM):
  maxima chains, partition functions and the singularity spectrum D(h).
- **Singularity analysis (microcanonical).** Per-pixel Hölder exponents from the gradient measure
  or from wavelet projections, the singularity spectrum by the histogram method, and
  reconstruction of the field from a band of exponents.
- **Multiscale edges.** Mallat–Zhong dyadic edges, Perona–Malik and complex cross-diffusion
  filtering, and a medial-axis extractor.
- **Decompositions.** Principal components of multi-band data, delay-embedded Tucker
  decomposition, and 2-D singular spectrum analysis.
- **Multiband data.** Sensor products such as ASTER import grouped by sensor and resolution.
  Each band can be analysed or removed on its own, and a stack displays as an RGB composite
  with per-band channel, solo and mute controls and ENVI-style stretches.
- **Band routing.** A bus assembles bands from several datasets on the same grid into one
  stack for the multi-band tools, and a band stack can follow the results of other layers
  as they change.
- **Anisotropy.** Gradient-direction arrows on the wavelet maxima, and the angle statistics
  of Arneodo, Decoster and Roux (2000), in image coordinates or as map bearings.
- **Filters on the results.** By scale, orientation, modulus, chain length and topology.
- **Regions of interest.** Any tool can run on a rectangular region. The region is read with a
  margin, so tools that need neighbouring data see real data beyond its edges.

Tools form a chain: transforms first, then filters. Changing a filter only redraws; changing a
transform recomputes, and every result is cached.

## Data

GeoTIFF (also inside a `.zip`), netCDF, HDF5 and HDF4 grids, and common image formats such as
PNG and JPEG.

## Install and run

GeoDynamix needs Python 3.12 and runs on Apple Silicon macOS and on Windows. Install it into an
environment of its own:

```bash
git clone https://github.com/MarioKartUltra/GeoDynamX.git
cd GeoDynamX
conda create -n geodynamix python=3.12
conda activate geodynamix
pip install -e ".[app]"
python -m dynamix.shell.app
```

## Compute settings

Every FFT runs on mlx (the GPU) on Apple Silicon, and on FFTW3 everywhere else. In Settings you
can choose the engine and the precision, 32 or 64 bit. 64-bit FFTs always run on FFTW3, because
mlx has no double-precision FFT. Changes take effect after a restart.

## Documentation

The [device reference](docs/reference/devices_reference.pdf) describes every tool and parameter,
multiband data and band routing, the wavelet normalisation, and the wavelet families, with
figures drawn from the application's own kernel code.

## Status

GeoDynamix is in beta: tools, parameters and the saved-project format may still change. Reading
EBSD files is not part of this release.

## License and citation

Copyright (C) 2026 Abraham Joseph Okayli Masaryk. GeoDynamix is free software under the GNU
General Public License; see [`LICENSE`](LICENSE). The author's code is GPL-2.0-or-later. The
modules translated from xsmurf (Decoster, Kestener, Roux and Arneodo; GPL-2.0) are GPL-2.0-only,
and the 1-D WTMM code is based on LastWave (Bacry and Mallat; GPL-2.0-or-later).
[`NOTICE`](NOTICE) lists the derived code and the principal scientific references, and
[`CITATION.cff`](CITATION.cff) explains how to cite this software.
