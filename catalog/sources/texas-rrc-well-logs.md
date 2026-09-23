---
id: texas-rrc-well-logs
name: Texas RRC Oil & Gas Well Records — GIS Well Logs
operator: "[[Railroad Commission of Texas]]"
portal: https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/well-information/
scale_tier: regional
resolution: "per-well logs and files"
data_types: [borehole]
regions: [north-america, united-states, texas]
coverage:
  name: Texas
  bbox: [-106.6, 25.8, -93.5, 36.5]
fetch:
  - method: manual
    url: https://www.rrc.texas.gov/resource-center/research/gis-viewer/
    notes: "Public GIS Map Viewer for oil, gas and pipeline data — no registration required; wells, leases, injection/disposal points"
  - method: manual
    url: https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/obtaining-commission-records/oil-gas-well-records-gis-well-logs/
    notes: "Scanned well logs (TIFF) served through the GIS viewer, plus monthly xlsx/xls inventory indices by district back to 2006"
  - method: api
    url: http://webapps2.rrc.state.tx.us/EWA/wellboreQueryAction.do
    notes: "EWA Wellbore Query — statewide well/API-number lookup; sibling EWA query tools cover proration, UIC injection wells, inactive and orphan wells"
formats: [tiff, xlsx]
license: "Public record (Texas Public Information Act / Natural Resources Code) — no open-data license statement; records retrieval requests may carry statutory fees"
added: 2026-08-11
verified: 2026-08-11
---

Texas's statewide well-record archive: scanned well logs (TIFF) reachable through the Public GIS
Map Viewer, monthly per-district inventory spreadsheets back to 2006, and a family of EWA query
tools (wellbore, proration, UIC, inactive/orphan wells) for tabular lookups by API number. No
registration wall for the viewer itself, but the RRC explicitly disclaims GIS-viewer positional
accuracy for excavation/pipeline-location purposes, and formal records requests can carry statutory
fees.
