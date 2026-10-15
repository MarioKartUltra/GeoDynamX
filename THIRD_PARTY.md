# Third-party code

External code enters GeoDynamix one way: a pinned tree under `third_party/<name>/` with its own
`LICENSE` and the upstream commit it was taken at, used through a thin adapter in `src/dynamix/devices/`
or `src/dynamix/core/`. Nothing external is copied into `src/` (the two exceptions predate this page and
are described in `NOTICE`: the xsmurf-derived modules and the LastWave port, both GPL).

Accepted licences, as GPL-2.0-or-later code may combine with them: MIT, BSD-2-Clause, BSD-3-Clause,
Apache-2.0 (with GPL-3.0-or-later only), LGPL-2.1-or-later, LGPL-3.0-or-later, GPL-2.0-only,
GPL-2.0-or-later, GPL-3.0-or-later. Code without a licence is never copied, only cited.

`tests/test_third_party_policy.py` checks that the rows below and the `third_party/` directories are the
same set and that each directory carries its `LICENSE`, and that `NOTICE` and the per-file SPDX headers
agree on which files are GPL-2.0-only.

| name | URL | commit | licence | path | used by |
|---|---|---|---|---|---|

None yet. The first candidate, the Torquato–Jiao code at `sdatkinson/TJ`, is a sphere-packing solver (GSL
and GLPK), not a lineal-path code; the lineal path is a first-party device (`src/dynamix/devices/lineal_path.py`).
