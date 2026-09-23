---
id: usgs-tnm-access
name: "USGS The National Map (TNM) Access API"
operator: "[[U.S. Geological Survey]]"
portal: https://apps.nationalmap.gov/downloader/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [dem, lidar]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://tnmaccess.nationalmap.gov/api/v1/products?bbox=-105,39,-104,40&max=1
    notes: >-
      Confirmed live: bbox product query returns real JSON (26,397 total matches for this
      test bbox, e.g. 3DHP FileGDB products). Sibling endpoint /api/v1/datasets lists the
      full dataset catalog fronted by this API (confirmed 2026-08-11): NED/3DEP DEM tiers
      (1m, 1/3 arc-sec, 1 arc-sec, Alaska IFSAR), 3DEP LiDAR Point Cloud (LPC) and OPR/IFSAR
      DEM, 3D Hydrography Program, NHD, WBD, US Topo, NAIP, GNIS, and small-scale datasets.
  - method: manual
    url: https://apps.nationalmap.gov/downloader/
    notes: "The National Map Downloader — confirmed live (HTTP 200); AOI + layer-picker UI over the same product catalog."
formats: [geotiff, img, las, laz, shapefile, geopackage]
license: "Public domain (U.S. Government work)"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** TNM Access is USGS's single products-search API fronting the entire
National Map catalog — this record is about the gateway itself, distinct from the
`usgs-3dep` entry elsewhere in this catalog, which records one specific DEM/LiDAR recipe
through the same endpoint. Flagship geology-relevant holdings reachable through it: the full
3DEP/NED elevation stack from 1-meter seamless DEM down to 1-arc-second national fallback
(`dem`), raw 3DEP LiDAR point clouds and Original Product Resolution (OPR) DEMs/IFSAR digital
surface models (`lidar`), plus non-elevation products (3D Hydrography, historical/current US
Topo quads, NAIP imagery, geographic names) that fall outside this catalog's data-type enum
but are worth knowing the same API serves them. No API key required for search; bulk
programmatic bbox/dataset/format queries are the main advantage over the downloader UI.
