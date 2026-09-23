---
id: california-cgs-fault-geology
name: California Geological Survey — Fault Activity Map & Geologic Map of California
operator: "[[California Geological Survey]]"
portal: https://www.conservation.ca.gov/cgs
scale_tier: regional
resolution: "1:750,000"
data_types: [structural-vectors, geologic-map]
regions: [north-america, united-states, california]
coverage:
  name: California
  bbox: [-124.5, 32.5, -114.1, 42.1]
fetch:
  - method: arcgis-rest
    url: https://gis.conservation.ca.gov/server/rest/services/CGS/FaultActivityMapCA/MapServer
    notes: 2010 Fault Activity Map — faults active in the Quaternary, digitized from original sources
  - method: arcgis-rest
    url: https://gis.conservation.ca.gov/server/rest/services/CGS/Geologic_Map_of_California/MapServer
license: "© California Dept. of Conservation — free public REST/export access; one service's iteminfo states \"All rights reserved\", redistribution terms otherwise unstated"
added: 2026-08-11
verified: 2026-08-11
---

CGS runs a large ArcGIS Server estate (gis.conservation.ca.gov/server/rest/services/CGS) — Fault
Activity Map and statewide Geologic Map are separate services on the same root, alongside historical
earthquakes, landslide inventories, and tsunami hazard layers. FAM traces are the Jennings/Bryant
compilation used as the state-authoritative fault layer; USGS Qfaults (also in this cell) is the
national analog it partly feeds into. License wording is inconsistent across services — some pages
claim "all rights reserved" despite open REST/download access, so treat downstream redistribution as
unclear rather than assuming public domain.
