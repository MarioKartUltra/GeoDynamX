---
id: denmark-dhm
name: Danmarks Højdemodel (DHM) — Denmark national LiDAR elevation model
operator: "[[Klimadatastyrelsen]]"
portal: https://www.klimadatastyrelsen.dk/groen-omstilling-og-klimasikring/klimasikring/danmarks-hoejdemodel
scale_tier: national
resolution: "0.4 m / 1 m / 5 m / 20 m DTM & DSM grids; ~4.5 pts/m2 LiDAR point cloud, ~5 cm vertical accuracy"
data_types: [dem, lidar]
regions: [europe, denmark]
coverage:
  name: Denmark
  bbox: [8.0, 54.5, 15.2, 57.8]
fetch:
  - method: wcs
    url: https://wcs.datafordeler.dk/DHMNedboer/dhm_wcs/1.0.0/WCS?service=WCS&request=GetCapabilities
    auth: api-key
    notes: "confirmed live (401 without key, i.e. real auth-gated service, not dead); free API key/OAuth via Datafordeler self-registration"
  - method: manual
    url: https://datafordeler.dk/dataoversigt/danmarks-hoejdemodel-dhm/dhm-wcs/
    notes: "Datafordeler service documentation/overview for DHM WCS (Terrain + Overflade) — confirmed live"
  - method: manual
    url: https://www.klimadatastyrelsen.dk/groen-omstilling-og-klimasikring/klimasikring/danmarks-hoejdemodel
    notes: "Klimadatastyrelsen (formerly SDFI/SDFE) landing page for the height-model programme — confirmed live"
formats: [geotiff, laz]
license: "Free reuse (Danish public-sector open data; free Datafordeler self-registration for the API key)"
added: 2026-08-11
verified: 2026-08-11
---

Denmark's ~415-billion-point national LiDAR dataset, delivered as DTM and DSM rasters at four grid
resolutions (0.4/1/5/20 m) plus the source point cloud, all through Datafordeler's WCS/WMS/download
services. The operating agency has been renamed twice in recent years — SDFE, then SDFI, now
Klimadatastyrelsen — and some older `dataforsyningen.dk`/`sdfi.dk` links now redirect or soft-404;
the URLs above were confirmed live directly. Access needs a free Datafordeler account and API key,
not anonymous download.
