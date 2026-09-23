---
id: sgc-colombia-geoportal
name: Servicio Geológico Colombiano — Geoportal & Open Data Portal
operator: "[[Servicio Geológico Colombiano]]"
portal: https://www2.sgc.gov.co/sgc/mapas/Paginas/geoportal.aspx
scale_tier: national
resolution: "1:100,000 to 1:1,000,000 (geologic map series); variable for geophysics/geochemistry/seismic catalogues"
data_types: [geologic-map, structural-vectors, gravity, magnetics, earthquake-catalog]
regions: [south-america, colombia]
coverage:
  name: Colombia
  bbox: [-79.0, -4.3, -66.8, 13.5]
fetch:
  - method: arcgis-rest
    url: https://srvags.sgc.gov.co/arcgis/rest/services?f=json
    notes: "confirmed live ArcGIS Server 10.91; folders incl. Geologia, Geological_Map_Colombia_2015, Geofisica, catalogo_sismos, Mapa_Metalogenico_2022"
  - method: http-download
    url: https://datos.sgc.gov.co/
    notes: "confirmed live Open Data portal (ArcGIS Hub); CSV/KML/Zip/GeoJSON/GeoTIFF downloads plus WMS/WFS API links"
formats: [shp, geojson, kml, geotiff, csv]
license: "Colombian open-government data (datos.gov.co framework) — no blanket CC statement found on the SGC portal itself; attribution to SGC customary"
added: 2026-08-11
verified: 2026-08-11
---

Colombia's national survey runs both a legacy ArcGIS Server (dozens of folders: geology, geophysics,
geochemistry atlases, metallogenic maps, and an earthquake catalogue) and a modern ArcGIS Hub open-data
portal with multi-format export (CSV/KML/GeoJSON/GeoTIFF) plus WMS/WFS API links. Spanish-language only.
No registration wall found on either the REST services root or the open-data portal.
