---
id: icgem-gravity-models
name: ICGEM — International Centre for Global Earth Models
operator: "[[GFZ German Research Centre for Geosciences]]"
portal: https://icgem.gfz.de/home
scale_tier: global
resolution: "spherical-harmonic models to ~degree/order 2190 (~5 arc-min); user-selectable output grid spacing"
data_types: [gravity]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://icgem.gfz.de/calcgrid
    notes: >-
      Web calculation service: pick a model (e.g. EGM2008, EIGEN-6C4, GOCE
      products), functional (geoid height, gravity anomaly, etc.), area and grid
      step; returns a computed grid. Companion services exist for user-defined
      point calculations and a 3D model browser (G3-Browser).
  - method: http-download
    url: https://icgem.gfz.de/tom_longtime
    notes: archive/index of the raw spherical-harmonic coefficient model files themselves
formats: [gfc, ascii-grid, netcdf]
native_metadata:
  - format: citation
    url: https://doi.org/10.5194/essd-11-647-2019
license: model-dependent; ICGEM archive itself is free access (per-model terms shown on each model's page)
added: 2026-08-11
verified: 2026-08-11
---

Archive-and-compute service for essentially all published global gravity field models
(static, time-variable GRACE/GOCE series, topographic, and even lunar/Venus/Mars models),
operated by GFZ under IAG/IGFS. Rather than one fixed grid, this is the place to generate
a gravity/geoid grid for a chosen model, area and resolution on demand — useful when a
specific epoch or functional (disturbance vs. anomaly vs. geoid height) is needed that a
static product like WGM2012 doesn't offer.
