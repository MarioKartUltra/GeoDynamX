---
id: wgm2012-gravity
name: WGM2012 — World Gravity Map
operator: "[[Bureau Gravimétrique International]]"
portal: https://bgi.obs-mip.fr/grids-and-models-2/grids-and-models-2-2/
scale_tier: global
resolution: 2 arc-min grid (terrain corrections from 1 arc-min ETOPO1)
data_types: [gravity]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://doi.org/10.18168/bgi.23
    notes: >-
      DOI resolves to the BGI catalogue record (bgi.obs-mip.fr/catalogue) for
      direct download of the complete Bouguer, isostatic (Airy-Heiskanen Tc=30km)
      and surface free-air grids plus ETOPO1-derived gravity disturbance/topography.
  - method: manual
    url: https://bgi.obs-mip.fr/data-products/outils/wgm2012-visualization-extraction/
    notes: web-based visualization/extraction tool for sub-area grid pulls
formats: [geotiff, netcdf]
license: >-
  academic/research use; BGI disclaims warranty on accuracy and accepts no
  responsibility for consequences of use (per WGM2012 terms of use)
added: 2026-08-11
verified: 2026-08-11
---

First release of global gravity anomalies (complete Bouguer, isostatic, free-air) computed
in spherical geometry, built from EGM2008 + DTU10 with ETOPO1 terrain corrections. Realized
by BGI with CGMW/UNESCO/IAG/IUGG/IUGS. Data is free to use but comes with an explicit
no-warranty clause — treat absolute values with the same caution as the source models.
