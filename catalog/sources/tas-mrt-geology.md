---
id: tas-mrt-geology
name: Mineral Resources Tasmania — Digital Geology
operator: "[[Mineral Resources Tasmania]]"
portal: https://www.thelist.tas.gov.au/app/content/home
scale_tier: regional
resolution: "1:25,000 (~50% coverage), 1:250,000, 1:500,000"
data_types: [structural-vectors, geologic-map]
regions: [australia, tasmania]
coverage:
  name: Tasmania
  bbox: [143.8, -43.7, 148.5, -39.5]
fetch:
  - method: arcgis-rest
    url: https://data.stategrowth.tas.gov.au/ags/rest/services/MRT/Geology_Tasmania/MapServer
    notes: >-
      Geology units, alteration, faults, contacts, linears, structure readings
      and outcrops layers across all three compiled scales. Confirmed live
      (HTTP 200, valid ArcGIS Server JSON).
formats: [shp, fgdb, esri-mapservice]
license: CC BY 3.0 AU
added: 2026-08-11
verified: 2026-08-11
---

Digital geology compiled at three scales, each with up to seven layers including faults,
contacts, linears and structure readings; 1:25,000 currently covers roughly half the state.
MRT's own product page (mrt.tas.gov.au/products/digital_data) sits behind a Cloudflare JS
challenge that blocked every curl/WebFetch attempt here — genuinely live for browsers, just
not scriptable — so this entry points instead at the underlying ArcGIS Server
(data.stategrowth.tas.gov.au) and theLIST, Tasmania's spatial portal, both confirmed live.
