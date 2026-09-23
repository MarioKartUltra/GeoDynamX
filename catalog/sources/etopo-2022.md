---
id: etopo-2022
name: ETOPO 2022 Global Relief Model
operator: "[[National Oceanic and Atmospheric Administration]]"
portal: https://www.ncei.noaa.gov/products/etopo-global-relief-model
scale_tier: global
resolution: "15/30/60 arc-second grids (~450 m / 900 m / 1.8 km at equator)"
data_types: [dem, bathymetry]
regions: [global]
coverage:
  name: Global (land + ocean)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/
    notes: >-
      Directory tree by resolution (15s/30s/60s) then by layer (elevation, sourceID,
      geoid); Ice-Surface and Bedrock versions; GeoTIFF and NetCDF. No login; directory
      listing confirmed live 2026-08-11.
  - method: manual
    url: https://www.ncei.noaa.gov/maps/grid-extract/
    notes: Interactive Grid Extract tool for custom-bbox subsets.
formats: [geotiff, netcdf]
license: Public domain (US Government work)
added: 2026-08-11
verified: 2026-08-11
---

NOAA's current global relief model, replacing ETOPO1, at three fixed resolutions with
separate Ice-Surface and Bedrock (sub-ice) variants for Greenland/Antarctica. Land
topography and ocean bathymetry are stitched from heterogeneous source grids (SRTM,
GEBCO-class bathymetry, national LiDAR, etc.), tracked per-cell in a companion sourceID
layer — worth checking before assuming uniform accuracy anywhere resolution matters.
