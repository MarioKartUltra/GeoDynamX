---
id: fabdem-v1-2
name: FABDEM V1-2 (Forest And Buildings removed Copernicus DEM)
operator: "[[University of Bristol]]"
portal: https://research-information.bris.ac.uk/en/datasets/fabdem-v1-2/
scale_tier: global
resolution: "30 m posting (1 arc-second)"
data_types: [dem]
regions: [global]
coverage:
  name: Global land, excluding Greenland and Antarctica
  bbox: [-180.0, -60.0, 180.0, 84.0]
fetch:
  - method: manual
    url: https://research-information.bris.ac.uk/en/datasets/fabdem-v1-2/
    notes: >-
      Pure repository record (verified live 2026-08-11), DOI
      10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn. The DOI 302-redirects to the data.bris.ac.uk
      CKAN file host (per-tile GeoTIFF zips + a GeoJSON tile index); that host reset every
      connection attempt during verification — confirm it is back up before relying on it.
      A commercial version with API/portal access is sold separately via Fathom
      (https://www.fathom.global/product/fabdem/, also verified live).
formats: [geotiff]
license: CC BY-NC-SA 4.0 (commercial use requires contacting the authors)
added: 2026-08-11
verified: 2026-08-11
---

Post-processes Copernicus GLO-30 to strip forest-canopy and building-height bias,
yielding a bare-earth surface rather than GLO-30's raw first-return DSM — useful
wherever vegetation or urban structures would otherwise bias slope/curvature analysis.
Non-commercial license only. The academic file host (data.bris.ac.uk) was unreachable at
verification time even though the DOI/record page resolves cleanly; treat the actual
download step as unconfirmed.
