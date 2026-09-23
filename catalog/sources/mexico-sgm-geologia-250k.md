---
id: mexico-sgm-geologia-250k
name: SGM — Cartografía Geológica de México, 1:250,000 (Continuo Nacional)
operator: "[[Servicio Geológico Mexicano]]"
portal: https://www.sgm.gob.mx/GeoInfoMexGobMx/
scale_tier: national
resolution: "1:250,000 (100% national coverage); 1:50,000 in priority mining districts"
data_types: [geologic-map]
regions: [north-america, mexico]
coverage:
  name: Mexico
  bbox: [-118.4, 14.5, -86.7, 32.7]
fetch:
  - method: manual
    url: https://www.sgm.gob.mx/GeoInfoMexGobMx/
    notes: interactive GeoInfoMex viewer; supports on-demand SHP export of lithology/geochemistry/magnetics layers by area
  - method: http-download
    url: http://mapserver.sgm.gob.mx/DatosAbiertosSGM/InformacionGeologica/Geologia_SGM.kml
    notes: "static national KML mirror of the \"Continuo Nacional de Geología\" (1995/2005 series, SGM-INEGI agreement), ~515MB, verified live"
license: Datos Abiertos (Gobierno de México) — free/open use; no explicit per-resource license text found
added: 2026-08-11
verified: 2026-08-11
---

Mexico's only nationwide-seamless lithology layer, built from the SGM/INEGI "Continuo Nacional"
program. The static KML mirror is large and unattributed on the format axis — it was not opened to
confirm whether fault/fold traces are digitized as a separate layer (Mexican 1:250k geologic quads
conventionally show them as part of the geology sheet, but that wasn't independently verified here),
so this entry is tagged geologic-map only. GeoInfoMex's on-demand export tool is the better route to
individual attributed layers (geochemistry, magnetics, lithology) if a structural layer is confirmed
in a later pass.
