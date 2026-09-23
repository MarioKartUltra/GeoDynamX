---
id: usgs-anss-comcat
name: USGS ANSS ComCat — Earthquake Catalog
operator: "[[USGS]]"
portal: https://earthquake.usgs.gov/data/comcat/
scale_tier: global
resolution: "event catalog; global M>=4.5 essentially complete since 1990s, denser/lower-magnitude coverage in well-instrumented regions and recent years"
data_types: [earthquake-catalog]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://earthquake.usgs.gov/fdsnws/event/1/query
    notes: >-
      FDSN event web service; live-tested with
      "?format=geojson&limit=1" — returns real-time event JSON. Supports
      geojson, csv, quakeml, text formats and full spatial/temporal/magnitude
      filtering.
formats: ["geojson", "csv", "quakeml", "text"]
native_metadata:
  - format: html
    url: https://earthquake.usgs.gov/data/comcat/data-eventterms.php
license: public domain (US federal); contributing networks may apply their own attribution norms
added: 2026-08-11
verified: 2026-08-11
---

The Advanced National Seismic System Comprehensive Catalog — USGS's aggregation of
authoritative earthquake parameters (hypocenters, magnitudes, moment tensors, products)
from contributing networks worldwide, since it is the ANSS's merged product rather than a
single network's picks. FDSN web service is fully open, no auth, and directly queryable —
the friction-free earthquake source in this cell.
