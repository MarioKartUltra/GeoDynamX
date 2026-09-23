---
id: noaa-ncei-multibeam-bathymetry
name: NOAA NCEI Multibeam Bathymetry Database
operator: "[[NOAA NCEI]]"
portal: https://www.ncei.noaa.gov/products/seafloor-mapping
scale_tier: global
resolution: "per-cruise raw swath (typically 1-100 m depending on platform/depth); archive of source surveys, not a merged grid"
data_types: [bathymetry]
regions: [global]
coverage:
  name: Global oceans (survey-track coverage, not continuous)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://data.ngdc.noaa.gov/platforms/ocean/ships/
    notes: >-
      Live directory tree, ship-name then cruise-ID folders (verified 2026-08-11: e.g.
      atlantis/, akademik_tryoshnikov/, baruna_jaya_iv/ — international contributions,
      not US-only). Raw/processed multibeam in native formats (.all, .mb58/.mb88 etc.,
      MB-System-readable) plus per-cruise metadata; no bulk global download, one cruise
      at a time.
  - method: manual
    url: https://www.ncei.noaa.gov/maps/bathymetry/
    notes: Bathymetric Data Viewer — map search across multibeam, singlebeam, lidar and crowdsourced holdings; links out to per-survey download.
formats: [mb-system, geotiff, xyz]
license: Public domain (US federal); international contributor surveys generally open, check per-cruise metadata
added: 2026-08-11
verified: 2026-08-11
---

The source-level archive behind the compiled global grids (GEBCO/GMRT/ETOPO already in this
catalog): individual ship cruises, contributed by US and international vessels, in
processing-software-native formats rather than a merged product. This is where to go for a
specific survey's original soundings and metadata; resolution and quality vary enormously
cruise-to-cruise, and there is no single bbox-clip API — the ships/ tree is browsed one
cruise folder at a time, or discovered via the map viewer.
