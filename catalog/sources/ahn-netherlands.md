---
id: ahn-netherlands
name: AHN — Actueel Hoogtebestand Nederland
operator: "[[Kadaster]]"
portal: https://www.ahn.nl/
scale_tier: national
resolution: "0.5 m DTM/DSM grid; point cloud ~6-10 pts/m2 (AHN4, 2020-2022 acquisition; AHN5/6 rolling out)"
data_types: [dem, lidar]
regions: [europe, netherlands]
coverage:
  name: Netherlands
  bbox: [3.3, 50.7, 7.3, 53.6]
fetch:
  - method: wcs
    url: https://www.pdok.nl/introductie/-/article/actueel-hoogtebestand-nederland-ahn
    notes: "PDOK dataset page listing the AHN WMS/WCS service endpoints (0.5 m DTM + DSM rasters) — confirmed live"
  - method: http-download
    url: https://www.pdok.nl/atom-downloadservices/-/article/actueel-hoogtebestand-nederland-ahn
    notes: "ATOM download service, per-map-sheet GeoTIFF/LAZ — confirmed live"
  - method: manual
    url: https://www.ahn.nl/
    notes: "AHN viewer, point-cloud viewer, and dataroom (incl. Caribbean Netherlands) — confirmed live"
formats: [geotiff, laz]
license: "Open data (Dutch public-sector open geodata policy; no reuse restrictions found on PDOK)"
added: 2026-08-11
verified: 2026-08-11
---

A consortium product (water authorities, provinces, Rijkswaterstaat) published nationally through
Kadaster's PDOK platform: wall-to-wall 0.5 m DTM and DSM plus the full classified point cloud
behind them, currently AHN4 (2020-2022 flights) with AHN5/6 rolling out region by region. WCS/WMS
give programmatic raster access; the ATOM service is the bulk per-sheet download path; the AHN
viewer is the friendliest way to check which version covers a given tile before pulling.
