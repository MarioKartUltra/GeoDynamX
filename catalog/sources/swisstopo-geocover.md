---
id: swisstopo-geocover
name: swisstopo GeoCover — Swiss geology, tectonics, gravity, aeromagnetics, DEM
operator: "[[Federal Office of Topography (swisstopo)]]"
portal: https://www.swisstopo.admin.ch/en/geological-model-2d-geocover
scale_tier: national
resolution: "1:25,000 (GeoCover)"
data_types: [structural-vectors, geologic-map, dem, gravity, magnetics]
regions: [europe, switzerland]
coverage:
  name: Switzerland
  bbox: [5.9, 45.8, 10.5, 47.9]
fetch:
  - method: wms
    url: https://wms.geo.admin.ch/?service=WMS&version=1.3.0&request=GetCapabilities&layers=ch.swisstopo.geologie-geocover
    notes: "GetCapabilities confirmed live (2.7MB); relevant layers: ch.swisstopo.geologie-geocover, ch.swisstopo.geologie-tektonische_karte, ch.swisstopo.geologie-gravimetrischer_atlas, ch.swisstopo.geologie-geophysik-aeromagnetische_karte_schweiz, ch.swisstopo.swissalti3d-reliefschattierung"
  - method: http-download
    url: https://www.swisstopo.admin.ch/en/geological-model-2d-geocover
    notes: "swissGEOCOVER2D as File Geodatabase (with ArcMap/QGIS project styling) or GeoPackage; whole-country vector geology at 1:25,000"
  - method: manual
    url: https://map.geo.admin.ch
    notes: "federal map.geo.admin.ch viewer exposes 100+ geology-relevant layers for browse/query/download"
license: "Free geodata (swisstopo terms of use for free geodata and geoservices)"
added: 2026-08-11
verified: 2026-08-11
---

Switzerland's national geoportal happens to be a genuine one-stop shop across families: GeoCover (2D
vector bedrock geology at 1:25,000, currently being re-surveyed through 2030), a dedicated tectonic map
layer, the gravimetric atlas, an aeromagnetic map of the whole country, and swissALTI3D high-resolution
elevation — all confirmed live in one WMS GetCapabilities response. GeoCover vector downloads ship with
ready-made ArcMap/QGIS symbology projects. Everything checked here is free; no registration wall found.
