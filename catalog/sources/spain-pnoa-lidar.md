---
id: spain-pnoa-lidar
name: PNOA-LiDAR — Spain national airborne LiDAR programme
operator: "[[Instituto Geográfico Nacional (Spain)]]"
portal: https://pnoa.ign.es/web/portal/pnoa-lidar/presentacion
scale_tier: national
resolution: "Point cloud density by coverage cycle: 0.5 pts/m2 (1st cycle, 2009-2015) to 5 pts/m2 (3rd cycle, 2022-2025); vertical RMSE improved from ~40 cm to ~10 cm; DTM/DSM derivatives"
data_types: [dem, lidar]
regions: [europe, spain]
coverage:
  name: Spain (mainland, Balearic and Canary Islands)
  bbox: [-18.2, 27.6, 4.4, 43.8]
fetch:
  - method: manual
    url: https://centrodedescargas.cnig.es/CentroDescargas/index.jsp
    notes: "CNIG Centro de Descargas — confirmed live; catalogue/AOI search for PNOA-LiDAR point clouds and DTM tiles"
  - method: manual
    url: https://pnoa.ign.es/web/portal/pnoa-lidar/presentacion
    notes: "programme presentation page — confirmed live; describes coverage cycles and 3rd-cycle AI-based point classification"
formats: [laz, asc, geotiff]
license: "CC BY 4.0 (attribution to IGN/CNIG)"
added: 2026-08-11
verified: 2026-08-11
---

A three-generation national LiDAR programme run jointly by IGN and Spain's autonomous communities,
with density and precision climbing each cycle — the 2022-2025 sweep hits 5 pts/m2 with AI-assisted
classification into 20+ feature types. Distribution is entirely through the CNIG download centre's
catalogue/AOI picker (no direct WCS/WFS endpoint found); tiles follow the Spanish national grid
(MTN25/50). Coverage cycles overlap in time by region, so check which cycle covers a given AOI
before assuming the highest density is available everywhere.
