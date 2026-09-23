---
id: cprm-geosgb-brazil
name: GeoSGB — SGB/CPRM Geoscientific Information System (Brazil)
operator: "[[Serviço Geológico do Brasil]]"
portal: https://geosgb.sgb.gov.br/geosgb
scale_tier: national
resolution: "1:1,000,000 down to 1:100,000 depending on sheet"
data_types: [geologic-map, structural-vectors, gravity, magnetics]
regions: [south-america, brazil]
coverage:
  name: Brazil
  bbox: [-74.0, -33.8, -34.7, 5.3]
fetch:
  - method: manual
    url: https://geosgb.sgb.gov.br/geosgb
    notes: "confirmed live ArcGIS web app (redirects to /geosgb iframe); Downloads panel exports shapefile/KML, plus WMS and OneGeology (UNESCO) service links"
  - method: http-download
    url: https://opendata.sgb.gov.br/
    notes: "SGB open-data catalogue mirror of the same holdings"
formats: [shp, kml]
license: "Brazilian federal open-data policy (SGB/CPRM); no blanket CC license text found — attribution to SGB/CPRM customary"
added: 2026-08-11
verified: 2026-08-11
---

Brazil's federal geological survey (SGB, formerly CPRM) consolidates bedrock geology, structural/fault
lines, aerogeophysics (gravity, magnetics), geochemistry, and outcrop/borehole descriptions in one ArcGIS
web app. Data exports as shapefile/KML from an in-app Downloads panel; the survey also participates in
the OneGeology project. Portuguese-only interface; no registration wall hit during verification, but no
explicit open-data license statement was found on the app itself.
