---
id: usgs-sgmc-gems-geodatabase
name: USGS State Geologic Map Compilation (SGMC) — GeMS geodatabase
operator: "[[U.S. Geological Survey]]"
portal: https://ngmdb.usgs.gov/Prodesc/proddesc_119417.htm
scale_tier: national
resolution: "1:1,000,000 (source state maps 1:24,000-1:500,000)"
data_types: [structural-vectors, geologic-map]
regions: [north-america, united-states]
coverage:
  name: Conterminous United States
  bbox: [-125.0, 24.5, -66.9, 49.4]
fetch:
  - method: http-download
    url: https://ngmdb.usgs.gov/ngm-bin/gems_download.pl?id=4742&pid=119417
    notes: 2026 GeMS-schema data release (Horton, DOI 10.5066/P1A3DQZK), ~1.8GB file geodatabase
  - method: manual
    url: https://www.sciencebase.gov/catalog/item/5888bf4fe4b05ccb964bab9d
    notes: legacy v1.1 (2017) ScienceBase item, kept live alongside the GeMS update
formats: [gdb]
license: Public Domain (U.S. Government Work)
added: 2026-08-11
verified: 2026-08-11
---

A seamless conterminous-US compilation stitched from 48 State geologic maps. The geodatabase carries
three feature classes: SGMC_Geology (polygons), SGMC_Points, and SGMC_Structure (polyline faults and
folds) — one of the few sources in this cell with a genuine national structural-vector layer rather
than just a geology polygon. The 2026 release migrated the data to the GeMS schema and is now served
through NGMDB's own download script rather than the original ScienceBase ArcGIS REST endpoint, which
returned intermittent 503s during verification.
