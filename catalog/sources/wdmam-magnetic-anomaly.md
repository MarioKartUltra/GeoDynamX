---
id: wdmam-magnetic-anomaly
name: WDMAM — World Digital Magnetic Anomaly Map v2.2
operator: "[[IAGA / CGMW]]"
portal: https://www.wdmam.org/
scale_tier: global
resolution: 3 arc-min grid
data_types: [magnetics]
regions: [global]
coverage:
  name: Global (continental and oceanic)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://www.wdmam.org/
    auth: free-registration
    notes: >-
      Interactive Leaflet map (wdmam.org, React SPA) for browsing and extracting the
      grid/anomaly layers; site infrastructure (CSP references AWS Cognito
      cognito-idp.eu-west-3.amazonaws.com) indicates an account is used for data
      export, distinct from the free public map view. Provisional map also offered
      as JPEG; full ASCII grid distributed on request per the site's stated policy.
formats: [ascii-grid, jpeg]
license: free for scientific/public use with citation (Dyment et al. 2016; Choi et al. for v2.2); no SPDX-style license stated
added: 2026-08-11
verified: 2026-08-11
---

IAGA/CGMW joint project (successor to Korhonen et al. 2007 v1). Compiles lithospheric
magnetic anomalies over both continents and oceans — complements EMAG2's satellite/marine
compilation with an independently curated, community-evaluated grid. Access is through a
JS-rendered map app rather than a static file index; no bulk anonymous download link found
on the landing page itself.
