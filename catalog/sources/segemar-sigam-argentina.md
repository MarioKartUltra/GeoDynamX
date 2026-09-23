---
id: segemar-sigam-argentina
name: SIGAM — SEGEMAR Geological-Mining-Environmental Information System (Argentina)
operator: "[[SEGEMAR]]"
portal: https://sigam.segemar.gov.ar/
scale_tier: national
resolution: "1:250,000 to 1:1,000,000 (geologic map series); variable for geophysical charts"
data_types: [geologic-map, structural-vectors, gravity, magnetics]
regions: [south-america, argentina]
coverage:
  name: Argentina
  bbox: [-73.6, -55.1, -53.6, -21.8]
fetch:
  - method: manual
    url: https://sigam.segemar.gov.ar/
    notes: "confirmed live viewer + catalogue (geologic, geophysical, geochemical, geo-environmental, hazard charts); portal documents WMS/WFS/CSW interoperability but a direct GeoServer ows probe (geoserver/ows) 404'd — use the portal's own service links rather than guessing a base path"
formats: [shp, kml]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

SEGEMAR's SIGAM bundles Argentina's geological map series with minero-metallogenic, geochemical,
geo-environmental, hazard, and geophysical charts behind one Spanish-language viewer/catalogue, licensed
CC BY 4.0 with no registration observed. The portal advertises WMS/WFS/CSW access; a guessed GeoServer
endpoint returned 404 during verification, so treat the exact machine-readable service URL as
unconfirmed — reach it via the portal's own layer links rather than the base path used here.
