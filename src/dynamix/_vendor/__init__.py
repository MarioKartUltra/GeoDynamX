"""Vendored copies of the author's own analysis packages (GeoDynamix_Beta, 2026-09-22).

The beta ships without DynamiX's two editable, private-repo dependencies so peers can install
it from one repository: ``wtmm`` (1-D WTMM; copied whole) and ``wtmm_ebsd`` (2-D CWT,
partition functions, chain filters, tensor WTMM -- the EBSD modules are deliberately LEFT OUT:
no orix, no MATLAB, no EBSD reading in the beta).

Source: ~/projects/Creep/wavelet at commit d73ba61 (2026-09-22). Copies are VERBATIM except:
- absolute ``wtmm`` / ``wtmm_ebsd`` imports rewritten to ``dynamix._vendor.*`` (the import
  rewrite the project's copy rule permits), so these can never collide with editable installs
  of the originals in the same environment;
- ``wtmm_ebsd/__init__.py`` imports only the modules copied here;
- ``wtmm_ebsd/cwt2d.py`` guards its top-level ``import mlx.core`` (Windows has no mlx; the app
  then runs the transform on FFTW3 through dynamix.core.fft_policy).
"""
