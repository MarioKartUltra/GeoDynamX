---
id: nd-dmr-oil-gas-wells
name: North Dakota Oil & Gas Division — well data and GIS viewer
operator: "[[North Dakota Department of Mineral Resources]]"
portal: https://www.dmr.nd.gov/dmr/oilgas
scale_tier: regional
resolution: "per-well files; statewide GIS well-location layer"
data_types: [borehole]
regions: [north-america, united-states, north-dakota]
coverage:
  name: North Dakota
  bbox: [-104.05, 45.9, -96.55, 49.0]
fetch:
  - method: manual
    url: https://www.dmr.nd.gov/oilgas/findwellsvw.asp
    notes: "Well Search — filter by operator, field, section/township/range; hourly-updated well index"
  - method: manual
    url: https://gis.dmr.nd.gov/dmrpublicportal/apps/webappviewer/index.html?id=a2b071015113437aa8d5a842e32bb49f
    notes: "ND Oil & Gas GIS Viewer — pan/zoom map with a GIS Data Download option for the well-location layer"
license: "Public record; bulk/subscription tiers apply — Basic $100/yr, Premium $500/yr (rates as posted Jan 2026) for expanded data services beyond the free web tools"
added: 2026-08-11
verified: 2026-08-11
---

North Dakota's Oil and Gas Division exposes a free well-search index and a public GIS viewer with a
downloadable well-location layer. Deeper access — scout tickets, bulk well-file documents, and
premium data feeds — sits behind a paid subscription (Basic/Premium tiers, priced annually). Onshore
Williston Basin emphasis; the well index updates hourly per the division's own notice.
