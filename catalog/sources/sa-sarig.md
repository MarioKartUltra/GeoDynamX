---
id: sa-sarig
name: SARIG — South Australian Resources Information Gateway
operator: "[[Geological Survey of South Australia]]"
portal: https://map.sarig.sa.gov.au/
scale_tier: regional
resolution: "statewide compilation ~1:250,000; detailed areas to 1:25,000-1:100,000"
data_types: [structural-vectors, geologic-map, borehole]
regions: [australia, south-australia]
coverage:
  name: South Australia
  bbox: [129.0, -38.1, 141.0, -25.9]
fetch:
  - method: manual
    url: https://map.sarig.sa.gov.au/
    notes: >-
      Free, no login. Right-click a map layer name for its download/WMS/WFS
      prompt (per SARIG's own documentation). The catalog.sarig.sa.gov.au
      metadata catalogue and energymining.sa.gov.au documentation pages returned
      HTTP 403 to every scripted fetch attempted here (bot wall), even though the
      map application itself is open — could not verify a stable per-layer
      service URL directly.
formats: [shp, "mif/mid", kml, csv]
license: CC BY
added: 2026-08-11
verified: 2026-08-11
---

SARIG bundles SA's geology, drillhole, geophysical-survey-index and mineral/petroleum
tenure layers into one free, no-login map viewer. Per-layer WMS/WFS and single-dataset
download exist per the department's own docs, but the supporting metadata catalogue and
department pages block scripted access outright — this entry verifies the map application
itself; expect to click through it by hand rather than script against a stable endpoint.
