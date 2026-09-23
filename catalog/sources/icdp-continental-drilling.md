---
id: icdp-continental-drilling
name: ICDP — International Continental Scientific Drilling Program
operator: "[[International Continental Scientific Drilling Program]]"
portal: https://www.icdp-online.org/
scale_tier: global
resolution: "per-project core/log/lithology records"
data_types: [borehole, core-sample]
regions: [global]
coverage:
  name: Global (continental drilling project sites)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://www.icdp-online.org/support/data-samples/
    notes: "Describes mDIS (mobile Drilling Information System), the open-source (GPL 2.0) tool ICDP projects use to capture core/log/lithology metadata during drilling"
  - method: manual
    url: https://dataservices.gfz-potsdam.de/portal/?q=icdp
    notes: "GFZ Data Services portal search for ICDP holdings — confirmed live (HTTP 200); JS-rendered results list not fetchable statically. Direct /icdp/ subpath returns 403."
license: "Per-project: Operational Data Set restricted to science team during drilling/moratorium, then opens to the science community; no single blanket license found"
added: 2026-08-11
verified: 2026-08-11
---

ICDP funds and coordinates continental (onshore) scientific drilling projects worldwide; each
project's cores, logs and lithology data are captured in mDIS during operations and archived
per-project (frequently at GFZ Data Services) after a moratorium period. There's no single unified
catalog/download page — access is per-project, and public availability depends on that project's
moratorium status. Contact is dm@icdp-online.org for specifics.
