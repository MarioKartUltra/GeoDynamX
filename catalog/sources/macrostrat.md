---
id: macrostrat
name: Macrostrat Geologic Map API
operator: "[[Macrostrat]]"
portal: https://macrostrat.org/
scale_tier: global
resolution: "variable — aggregated from ~225 source geologic maps at native scales (roughly 1:24,000 to 1:1,000,000+)"
data_types: [geologic-map]
regions: [global]
coverage:
  name: Global (map density concentrated in North America and Australia)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://macrostrat.org/api/v2/geologic_units/map
    notes: >-
      GeoJSON polygons by bbox/lat-lng-z (tile-style) query params; format=geojson.
      /geologic_units/map/legend returns unit metadata (lithology, age); no dedicated
      fault/structural-line endpoint — this feeds the geologic-map half of the cell only.
formats: [geojson]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

A homogenized mosaic of ~225 regional/national geologic maps (~2.5M polygons) behind one
API, built and maintained out of UW-Madison. No account or key needed — plain HTTP GET
returns GeoJSON. The catch: source-map heterogeneity means scale and vintage vary polygon
to polygon, so a single bbox query can silently mix a 1:24k state map with a coarse
continental compilation. The `/columns` and `/paleogeography` routes are stratigraphic,
not spatial-structural — out of scope for this cell.
