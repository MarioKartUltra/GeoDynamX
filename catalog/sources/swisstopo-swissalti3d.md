---
id: swisstopo-swissalti3d
name: swisstopo swissALTI3D + swissSURFACE3D — Swiss DEM and LiDAR point cloud
operator: "[[Federal Office of Topography (swisstopo)]]"
portal: https://www.swisstopo.admin.ch/en/height-model-swissalti3d
scale_tier: national
resolution: "0.5 m / 2 m DEM grid (swissALTI3D); LiDAR point cloud 5-40 pts/m2 depending on region/acquisition cycle (swissSURFACE3D)"
data_types: [dem, lidar]
regions: [europe, switzerland]
coverage:
  name: Switzerland and Liechtenstein
  bbox: [5.9, 45.8, 10.5, 47.9]
fetch:
  - method: stac
    url: https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d
    notes: "STAC collection for swissALTI3D — confirmed live; ground sampling 0.5/2.0 m, native EPSG:2056"
  - method: manual
    url: https://www.swisstopo.admin.ch/en/height-model-swisssurface3d
    notes: "swissSURFACE3D classified LiDAR point cloud, LAZ 1.4/COPC (2024+) or LAS 1.2 (pre-2024), via the geo.admin.ch download portal — confirmed live"
  - method: manual
    url: https://www.swisstopo.admin.ch/en/height-model-swissalti3d
    notes: "product page: sample data, Cloud-Optimized GeoTIFF / XYZ ASCII / ESRI ASCII-on-request — confirmed live"
formats: [geotiff, laz, asc]
license: "Free geodata (swisstopo terms of use for free geodata and geoservices)"
added: 2026-08-11
verified: 2026-08-11
---

The elevation half of swisstopo's stack (the geology/gravity/magnetics half is already catalogued
under `swisstopo-geocover`): a vegetation-and-building-stripped bare-earth DEM at 0.5 m (dense
areas) or 2 m elsewhere, plus its companion classified point cloud swissSURFACE3D, whose density
and format changed with the 2024 acquisition cycle (LAS 1.2 zipped tiles before, cloud-optimized
COPC LAZ 1.4 after — check the download portal's date filter). Whole-country coverage, re-surveyed
on a ~6-year cycle; everything here is free, no registration wall found.
