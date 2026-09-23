---
id: nz-linz-elevation
name: LINZ National Elevation — New Zealand DEM + LiDAR
operator: "[[Land Information New Zealand]]"
portal: https://data.linz.govt.nz/group/national-elevation/layers/
scale_tier: national
resolution: "1 m DEM/DSM (LiDAR-derived, ~90% of mainland covered); national 8 m DEM fills gaps; source point clouds also published"
data_types: [dem, lidar]
regions: [oceania, new-zealand]
coverage:
  name: New Zealand (North and South Islands; LiDAR coverage patchy pre-2015, near-complete since)
  bbox: [166.3, -47.5, 178.6, -34.0]
fetch:
  - method: http-download
    url: https://data.linz.govt.nz/group/national-elevation/layers/
    notes: "LINZ Data Service — confirmed live; national-elevation group indexes every regional 1 m DEM/DSM/point-cloud survey layer plus the 8 m national DEM"
  - method: http-download
    url: https://registry.opendata.aws/nz-elevation/
    notes: "confirmed live; same data mirrored as an AWS Open Data bucket (Cloud-Optimized GeoTIFF), no LINZ account needed"
  - method: http-download
    url: https://data.linz.govt.nz/layer/51768-nz-8m-digital-elevation-model-2012/
    notes: "confirmed live; the coarser seamless national DEM used where 1 m LiDAR coverage doesn't yet exist"
formats: [geotiff]
license: "CC BY 4.0"
added: 2026-08-11
verified: 2026-08-11
---

New Zealand's open-data elevation programme is really dozens of separate regional survey layers
(each its own dataset/ID in the LDS catalogue) rather than one seamless national product — the 8 m
DEM is the only truly wall-to-wall layer, with 1 m LiDAR DEM/DSM and point clouds now covering
~90% of the mainland tile-by-tile. Cloud-Optimized GeoTIFFs use LERC compression, tiled on NZTM2000
in a 1:1,000 layout. The same data is mirrored to AWS Open Data, which is the easier bulk-pull path
if the per-layer LDS catalogue browsing gets tedious.
