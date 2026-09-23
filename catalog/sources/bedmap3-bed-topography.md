---
id: bedmap3-bed-topography
name: BEDMAP3 — Ice Thickness, Bed and Surface Elevation for Antarctica
operator: "[[British Antarctic Survey]]"
portal: https://www.bas.ac.uk/project/bedmap/
scale_tier: continental
resolution: "500 m gridded products; ~50M source survey points at native line resolution"
data_types: [dem]
regions: [antarctica]
coverage:
  name: Antarctica (south of 60°S)
  bbox: [-180.0, -90.0, 180.0, -60.0]
fetch:
  - method: http-download
    url: https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2
    notes: >-
      Gridding-products release: bed elevation, ice-thickness, surface elevation, uncertainty
      and data-density grids at 500 m in EPSG:3031, plus a streamline-point shapefile. GeoTIFF
      (7 files, ~2.1 GB), single netCDF (~2.4 GB), or shapefile. Hosted on BAS's RAMADDA
      repository (ramadda.data.bas.ac.uk).
formats: [geotiff, netcdf, shp]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

Successor to Bedmap1/2 — subglacial bed topography and ice thickness for all of Antarctica,
compiled from 84 new aerogeophysical surveys plus legacy data (17+15 providers). This is the
bedrock-under-ice DEM, not an ice-surface DEM — pair with a surface-elevation product (e.g.
REMA) if the surface, not the bed, is the analysis target. Large files: multi-GB per format.
