---
id: elvis-elevation
name: ELVIS — Elevation and Depth (Foundation Spatial Data Framework)
operator: "[[Geoscience Australia]]"
portal: https://elevation.fsdf.org.au/
scale_tier: national
resolution: "1-5 m DEM/DSM (LiDAR-derived, coverage patchy); raw point clouds in LAZ"
data_types: [dem, lidar]
regions: [australia]
coverage:
  name: Australia (LiDAR coverage concentrated coastal/urban/agricultural, sparse inland)
  bbox: [112.9, -43.7, 153.7, -9.1]
fetch:
  - method: manual
    url: https://elevation.fsdf.org.au/
    notes: >-
      Menu > Order Data, draw an AOI; free, no login required for ordering,
      15 GB/request cap; packaged download link is emailed once the extract is built.
formats: [geotiff, laz]
license: CC BY 4.0 (varies by contributing state/territory agency)
added: 2026-08-11
verified: 2026-08-11
---

The national clearinghouse for LiDAR-derived elevation and bathymetry, aggregating
state/territory LiDAR programmes behind one AOI-based order tool. There is no direct file
browsing or bulk API — every pull is a drawn polygon, queued server-side, and delivered as
an emailed link, capped at 15 GB/request. Coverage is dense over settled coastal Australia
and thin over the interior; check tile coverage in the viewer before committing to a region.
