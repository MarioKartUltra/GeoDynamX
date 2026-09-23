#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# Launch the DynamiX shell without installing the package.
#
#   scripts/shell.sh                              # docs/demo/dem_crop.npz
#   scripts/shell.sh path/to/field.npz            # any raster RasterField can read
#   scripts/shell.sh path/to/field.npz --render out.png   # headless frame, for verification
#
# Uses the xsmurf env if present, else whatever python3 is on PATH.
set -e
cd "$(dirname "$0")/.."

PY="$HOME/miniforge/envs/xsmurf/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

PYTHONPATH=src exec "$PY" -m dynamix.shell.app "$@"
