---
id: usgs-earthexplorer-m2m
name: "USGS EarthExplorer + M2M API"
operator: "[[U.S. Geological Survey]]"
portal: https://earthexplorer.usgs.gov/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [dem]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://m2m.cr.usgs.gov/api/api/json/stable/login-token
    auth: ers-account
    notes: >-
      Confirmed live: an unauthenticated POST returns structured JSON
      ({"errorCode":"INPUT_PARAMETER_REQUIRED","errorMessage":"...username is required..."})
      rather than a connection failure — the M2M JSON-RPC-style API is reachable. Full API
      docs themselves sit behind a USGS ERS login wall (m2m.cr.usgs.gov/api/docs/... redirects
      to ers.cr.usgs.gov/login) — this is real friction, not just a login-to-download step: the
      reference material is gated too, and M2M access is a separate grant on top of a basic ERS
      account.
  - method: manual
    url: https://earthexplorer.usgs.gov/
    notes: "EarthExplorer UI — confirmed live (HTTP 200). Digital Elevation category (SRTM, GTOPO30, etc.) confirmed via USGS EROS documentation."
formats: [geotiff, hdf, tar]
license: "Public domain (U.S. Government work); some partner/commercial collections carry separate terms"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** EarthExplorer (browse/order UI) and the M2M API (its machine-readable
sibling) are the two faces of USGS's central holdings search — one catalog fronting Landsat
(all generations), Sentinel-2 (USGS mirror), NAIP aerial imagery, declassified CORONA/hexagon
imagery, and — the geology-relevant slice this catalog cares about — the Digital Elevation
category (SRTM 1-arc-second global, legacy GTOPO30, and other historical DEM products; `dem`).
Both EarthExplorer and M2M require a free USGS ERS account; M2M specifically requires
requesting and being granted separate API access on top of that account, and even the M2M API
*documentation* is gated behind ERS login — confirmed by the live redirect chain above. Treat
this as the highest-friction gateway in this cell: real, live, and JSON-native once
authenticated, but not casually scriptable without first clearing the access-request step.
