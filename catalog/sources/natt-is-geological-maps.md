---
id: natt-is-geological-maps
name: Náttúrufræðistofnun — geological and tectonic maps of Iceland
operator: "[[Icelandic Institute of Natural History]]"
portal: https://www.natt.is/en/resources/geospatial-data/geological-maps
scale_tier: national
resolution: "1:600,000 (bedrock/tectonics overview); 1:250,000 (9-sheet bedrock series); 1:100,000 (selected regional sheets)"
data_types: [structural-vectors, geologic-map]
regions: [europe, iceland]
coverage:
  name: Iceland
  bbox: [-24.6, 63.3, -13.5, 66.6]
fetch:
  - method: manual
    url: https://natt.gis.is/mapview/?app=natt
    notes: "Geology of Iceland map viewer — confirmed live; bedrock geology 1:600,000 (2014), tectonics map 1:600,000 (2009, rift/fault zones), 1:250,000 nine-sheet series (7 of 9 published), regional 1:100,000 sheets"
  - method: manual
    url: https://gatt.gis.is/geonetwork/srv/ice/catalog.search
    notes: "metadata/download catalog (Lýsigagnagátt) — confirmed live; individual map records carry direct download links and full metadata"
license: "CC BY 4.0 — published under Iceland's public-sector information reuse law no. 45/2018; attribution to Náttúrufræðistofnun required"
added: 2026-08-11
verified: 2026-08-11
---

Iceland's national geological survey function sits with Náttúrufræðistofnun (Icelandic Institute of Natural
History), not the geothermal-consulting ÍSOR — ÍSOR was checked and ruled out as a data source (services
firm, no public data catalog). Náttúrufræðistofnun publishes bedrock geology at overview (1:600,000) and
regional (1:250,000, 1:100,000) scales, plus a dedicated 1:600,000 tectonics map carrying Iceland's rift
zones and fault systems — a genuine structural-vectors match. Access is via a map viewer and a GeoNetwork
metadata/download catalog rather than a raw WMS GetCapabilities URL; both confirmed live. Fully open, CC BY.
