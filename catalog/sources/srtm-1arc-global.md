---
id: srtm-1arc-global
name: SRTM 1 Arc-Second Global
operator: "[[United States Geological Survey]]"
portal: https://earthexplorer.usgs.gov/
scale_tier: global
resolution: "30 m posting (1 arc-second); 2x1 arc-second above 50°N / below 50°S"
data_types: [dem]
regions: [global]
coverage:
  name: Global land, 56°S-60°N
  bbox: [-180.0, -56.0, 180.0, 60.0]
fetch:
  - method: manual
    url: https://earthexplorer.usgs.gov/
    auth: free-registration
    notes: >-
      Search dataset category "Digital Elevation" > "SRTM 1 Arc-Second Global"; free
      USGS EROS login required before download of GeoTIFF/BIL/DTED tiles (confirmed:
      unauthenticated session shows "must be logged in to download").
formats: [geotiff, bil, dted]
license: Public domain (US Government work; NASA/USGS)
added: 2026-08-11
verified: 2026-08-11
---

The original Shuttle Radar Topography Mission C-band global DEM (2000), reprocessed as
v3. EarthExplorer is the standard USGS distribution point — free but login-gated, and
limited to 56°S-60°N (the shuttle's orbital inclination). Vertical datum EGM96; some
tiles still carry voids. Largely superseded by Copernicus GLO-30 and NASADEM for new
work, but remains the longest-running global elevation baseline and common reference
surface.
