---
id: planetary-computer-stac
name: "Microsoft Planetary Computer STAC API"
operator: "[[Microsoft]]"
portal: https://planetarycomputer.microsoft.com/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [dem, lidar]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: stac
    url: https://planetarycomputer.microsoft.com/api/stac/v1
    notes: >-
      Confirmed live: root catalog document returns real STAC JSON (id "microsoft-pc").
      /collections lists 135 collections (confirmed 2026-08-11), including cop-dem-glo-30,
      cop-dem-glo-90, alos-dem, nasadem (`dem`), and the full 3dep-lidar-* family
      (3dep-seamless, 3dep-lidar-dsm/dtm/hag/intensity/classification/returns/copc — `lidar`).
      cop-dem-glo-30 assets confirmed to resolve to a public Azure blob URL with no SAS token
      required; other collections (e.g. Sentinel-2) require a free short-lived SAS token from
      the Planetary Computer Hub for full-resolution asset access.
  - method: manual
    url: https://planetarycomputer.microsoft.com/
    notes: "Planetary Computer portal/Explorer — confirmed live (HTTP 200); browse and preview UI over the same STAC catalog."
formats: [geotiff, cog, zarr, parquet]
license: "Varies per collection — Copernicus DEM under the ESA/Airbus free license; USGS 3DEP products public domain; check per-collection license field"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** One STAC API fronting 135+ analysis-ready cloud-optimized collections
curated by Microsoft's Planetary Computer, mixing USGS, ESA, NASA, and other agency source
data behind a single search/query interface plus (for many collections) hosted compute.
Flagship geology-relevant holdings reachable through it: Copernicus DEM GLO-30/GLO-90 and
NASADEM/ALOS global elevation (`dem`), and the full USGS 3DEP LiDAR family — seamless DTM/DSM
mosaics plus point-cloud-derived intensity, classification, height-above-ground, and return
layers (`lidar`) — the same underlying 3DEP LiDAR the `usgs-tnm-access` and `usgs-3dep`
entries reach through USGS's own API, but pre-tiled as cloud-optimized GeoTIFF/COPC here.
Search and metadata access are free and unauthenticated; some collections require a free
short-lived SAS token (obtained via the Hub or a simple token-service call) for full-asset
access, others (confirmed for Copernicus DEM) are fully open with no token at all.
