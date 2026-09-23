---
id: usgs-3dep
name: USGS 3D Elevation Program (3DEP) — seamless DEMs + LiDAR point clouds
operator: "[[U.S. Geological Survey]]"
portal: https://www.usgs.gov/3d-elevation-program
scale_tier: national
resolution: "Seamless 1 m DEM (growing CONUS coverage) and 1/3 arc-second (~10 m) national seamless DEM; raw LiDAR point clouds 2-20+ pts/m2, project-dependent"
data_types: [dem, lidar]
regions: [north-america, united-states]
coverage:
  name: United States (CONUS, Alaska, Hawaii, territories — LiDAR coverage tiers vary by project)
  bbox: [-179.9, 18.0, -66.9, 71.5]
fetch:
  - method: api
    url: https://tnmaccess.nationalmap.gov/api/v1/products?prodFormats=LAS,LAZ&max=1
    notes: "TNM Access API — confirmed live (JSON); query by bbox/dataset/format for direct product URLs"
  - method: manual
    url: https://apps.nationalmap.gov/downloader/
    notes: "The National Map Downloader — confirmed live; AOI + layer picker (1m DEM, 3DEP LiDAR Point Cloud, 1/3 arc-second DEM, etc.)"
  - method: http-download
    url: https://s3-us-west-2.amazonaws.com/usgs-lidar-public/
    notes: "public AWS bucket — confirmed live; per-project Entwine Point Tile (EPT) LiDAR, no auth"
formats: [geotiff, laz, img]
license: "Public domain (U.S. Government work)"
added: 2026-08-11
verified: 2026-08-11
---

The national LiDAR/DEM backbone: seamless 1 m DEM where flown, a national 1/3 arc-second (~10 m)
fallback everywhere else, and raw point clouds staged both through The National Map and as
public, unauthenticated Entwine tiles on AWS — the latter is the easiest bulk-programmatic path
(no API key, no order queue). TNM Access API gives scriptable bbox queries; the web downloader is
better for one-off AOI pulls. Coverage tiers (1 m vs 10 m vs point cloud) vary project-to-project,
so check the downloader's footprint layer before assuming 1 m is available everywhere.
