---
id: usgs-quaternary-faults
name: USGS Quaternary Fault and Fold Database of the United States
operator: "[[U.S. Geological Survey]]"
portal: https://www.usgs.gov/programs/earthquake-hazards/faults
scale_tier: national
resolution: "compiled from source mapping at 1:24,000-1:250,000"
data_types: [structural-vectors]
regions: [north-america, united-states]
coverage:
  name: United States (incl. Alaska, Hawaii, Puerto Rico)
  bbox: [-179.15, 17.9, -65.2, 71.44]
fetch:
  - method: http-download
    url: https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip
    notes: statewide-seamless fault/fold traces, shapefile, ~32MB
  - method: manual
    url: https://doi.org/10.5066/F7S75FJM
    notes: interactive fault map; the old text-search "Database Search" tool was retired Feb 26 2026 in favor of this map
formats: [shp, kmz]
license: Public Domain (U.S. Government Work)
added: 2026-08-11
verified: 2026-08-11
---

Fault and fold traces judged active in the Quaternary, each carrying a legacy report with slip rate,
recency, and paleoseismic notes — the reference structural-vector layer nearly every state hazard map
builds on (Nevada and California entries in this cell explicitly cross-reference it). Pure line data,
no accompanying geology polygons, so pair it with a state or SGMC geologic map for full context.
