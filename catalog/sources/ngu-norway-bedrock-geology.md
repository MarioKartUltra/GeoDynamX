---
id: ngu-norway-bedrock-geology
name: NGU Geological Datasets — bedrock, structural elements, boreholes (Norway)
operator: "[[Geological Survey of Norway]]"
portal: https://www.ngu.no/en
scale_tier: national
resolution: "1:50,000 (N50) to 1:1,350,000 (N1350)"
data_types: [geologic-map, structural-vectors, borehole]
regions: [europe, norway]
coverage:
  name: Norway
  bbox: [4.5, 57.9, 31.2, 71.2]
fetch:
  - method: http-download
    url: https://geo.ngu.no/download/order?lang=en
    notes: "confirmed live order page; select a dataset (bedrock maps N50/N250/N1350, weakness zones, groundwater boreholes, etc.) and UTM zone, download SOSI/Shape/FGDB"
  - method: wms
    url: https://www.ngu.no/en/taxonomy/term/36
    notes: "index of NGU's WMS services (bedrock, airborne/ground geophysics, boreholes, petrophysics, gravimetry), served through the Norway Digital SDI"
formats: [sosi, shp, fgdb]
license: "NLOD (Norwegian licence for open government data) — attribution required"
added: 2026-08-11
verified: 2026-08-11
---

Norway's national geological survey packages bedrock geology at three fixed scales plus structural
"weakness zones" and groundwater borehole datasets through a self-service order page (pick dataset + UTM
zone, download SOSI/Shape/FGDB). Gravity and magnetics are not on the download-order page itself but are
listed among NGU's ~17 WMS services for live viewing. Free under NLOD; mandatory attribution string is
specified on the download page.
