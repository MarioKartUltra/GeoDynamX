---
id: uk-ea-lidar
name: Environment Agency National LIDAR Programme — England DTM/DSM + point cloud
operator: "[[Environment Agency]]"
portal: https://environment.data.gov.uk/survey
scale_tier: national
resolution: "Composite DTM at 1 m / 2 m / 10 m; composite DSM at 2 m; underlying yearly survey tiles (point cloud + raster) at finer native resolution by project, vertical accuracy ±15 cm RMSE"
data_types: [dem, lidar]
regions: [europe, united-kingdom, england]
coverage:
  name: England (~99% coverage)
  bbox: [-6.5, 49.9, 1.8, 55.8]
fetch:
  - method: manual
    url: https://environment.data.gov.uk/survey
    notes: "Defra Survey Data Download — AOI/tile picker for the raw National LiDAR Programme survey tiles (DTM, DSM, point cloud, intensity); confirmed live"
  - method: http-download
    url: https://www.data.gov.uk/dataset/01b3ee39-da3f-47b6-83da-dc98e73a461f/lidar-composite-digital-terrain-model-dtm-1m
    notes: "seamless composite DTM, 1 m, GeoTIFF, 5 km tiles on the OS National Grid — confirmed live"
  - method: manual
    url: https://www.data.gov.uk/dataset/f0db0249-f17b-4036-9e65-309148c97ce4/national-lidar-programme
    notes: "National LiDAR Programme parent record, linking out to the DTM/DSM/point-cloud/intensity composite products — confirmed live"
formats: [geotiff, laz, asc]
license: "Open Government Licence v3.0"
added: 2026-08-11
verified: 2026-08-11
---

Two layers of the same programme: a merged, seamless composite DTM/DSM (1-10 m GeoTIFF, resampled
from the best available survey each tile) for quick coverage, and the underlying yearly National
LiDAR Programme survey tiles — original point clouds and rasters — for anyone who needs the source
data rather than the composite. Both are free with no registration wall. Tiling is on the OS
National Grid in 5 km blocks; heights are OSGM/Newlyn with the OSTN15 transformation, worth
checking before mixing with WGS84-native sources.
