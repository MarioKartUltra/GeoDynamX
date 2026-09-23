---
id: nrcan-bedrock-geology-index
name: NRCan/GSC — Bedrock Geology (national index & generalized compilation)
operator: "[[Geological Survey of Canada]]"
portal: https://open.canada.ca/data/en/dataset/5887347a-f64d-c5e6-b983-8cace169313b
scale_tier: national
resolution: "generalized national polygon; index of CGM/A-series/preliminary sheets at native scales (1:20,000-1:1,000,000)"
data_types: [geologic-map]
regions: [north-america, canada]
coverage:
  name: Canada
  bbox: [-141.0, 41.7, -52.6, 83.1]
fetch:
  - method: arcgis-rest
    url: https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/gsc_bedrock_geology_en/MapServer
    notes: layers are "Map index" (footprints of published Bedrock/CGM sheets) and "Bedrock geology" (generalized polygon)
  - method: http-download
    url: https://ftp.maps.canada.ca/pub/nrcan_rncan/Geology_Geologie/bedrock_geology-geologie_socle_rocheux/gsc_bedrock_geology.gdb.zip
    notes: file geodatabase backing the service above, updated 2024-07-26
license: Open Government Licence - Canada
added: 2026-08-11
verified: 2026-08-11
---

Replaces the originally-proposed GEOSCAN as the Canada national entry: `geoscan.nrcan.gc.ca` no
longer resolves (NXDOMAIN), and GEOSCAN was a publication/report bibliography rather than a geospatial
source regardless. This service is actively maintained (2024 data) but is an index/generalized layer —
the downloaded shapefile bundle contains only one polygon feature class ("National"), no distinct
fault/structure line layer, so it is tagged geologic-map only. A 1996 1:5,000,000 "Bedrock Geological
Map of Canada" (DOI 10.4095/208175) explicitly depicts major faults but is a "Historical Archive" with
no verifiable direct GIS download in this pass; Ontario's OGS entry in this cell is the more reliable
Canadian structural-vector source for now.
