---
id: gmrt
name: Global Multi-Resolution Topography (GMRT) Synthesis
operator: "[[Lamont-Doherty Earth Observatory]]"
portal: https://www.gmrt.org/
scale_tier: global
resolution: "up to ~100 m grid-node spacing (multibeam-derived; coarser where source surveys are sparse)"
data_types: [bathymetry, dem]
regions: [global]
coverage:
  name: Global (land + ocean)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://www.gmrt.org/services/GridServer?north={maxlat}&south={minlat}&east={maxlon}&west={minlon}&layer=topo&format=geotiff
    notes: >-
      REST GridServer endpoint; verified live 2026-08-11 (sample bbox query
      north=-2.9928&west=-90&east=-69.2578&south=-19.4825 returned a working 1.1 MB
      GeoTIFF, no auth). Also serves netCDF (GMT3/COARDS) and Esri ASCII; up to ~20x20
      deg at max (~100 m) resolution, 2 GB cap per request.
  - method: manual
    url: https://www.gmrt.org/GMRTMapTool/
    notes: Interactive MapTool with standard, north-polar and south-polar projections.
formats: [geotiff, netcdf, esri-ascii]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

Continually-updated synthesis (v4.5.0, 2026) of curated multibeam bathymetry (1,644+
surveys) merged with terrestrial DEMs, maintained by LDEO/Columbia with NSF funding — the
grid behind Google Earth's ocean floor and folded into GEBCO since 2014. GridServer is
script-friendly and unauthenticated. Resolution tracks survey density hard: open-ocean
cells far from ship tracks fall back to coarse satellite-derived bathymetry under the
same "topo" layer, so a bbox request can silently mix ~100 m and multi-km effective
resolution.
