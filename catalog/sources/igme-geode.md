---
id: igme-geode
name: IGME GEODE — continuous digital geological map of Spain
operator: "[[Instituto Geológico y Minero de España]]"
portal: https://info.igme.es/cartografiadigital/geologica/Geode.aspx?language=en
scale_tier: national
resolution: "1:50,000 (peninsula); 1:25,000 (insular territories)"
data_types: [structural-vectors, geologic-map]
regions: [europe, spain]
coverage:
  name: Spain (peninsula, Balearics, Canary Islands)
  bbox: [-18.2, 27.6, 4.3, 43.8]
fetch:
  - method: wms
    url: https://mapas.igme.es/gis/services/Cartografia_Geologica/IGME_Geode_50/MapServer/WmsServer?service=WMS&request=GetCapabilities
    notes: "GetCapabilities confirmed live (valid WMS_Capabilities XML, ~36KB)"
  - method: arcgis-rest
    url: https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geode_50/MapServer
    notes: "ArcGIS Server REST endpoint for the same GEODE 1:50,000 service"
  - method: manual
    url: https://info.igme.es/cartografiadigital/geologica/Geode.aspx?language=en
    notes: "full-country vector shapefile is not an instant download — request via cartografiadigital@igme.es, priced by surface area covered; WMS/WmsServer above is the no-request path"
license: "Free, general-purpose IGME data licence (commercial and non-commercial reuse permitted, attribution to IGME required); vector shapefile fulfilment is a priced/request process distinct from the licence itself"
added: 2026-08-11
verified: 2026-08-11
---

GEODE homogenizes Spain's older MAGNA 1:50,000 sheet series (1972-2003) into a seamless, cartographically
continuous digital geological map, with chronostratigraphic units and the contacts/faults between them.
The WMS and ArcGIS REST endpoints are open and instant; the one real friction point is that a full-country
vector shapefile isn't a self-serve download — IGME fulfils it on request, priced by area, even though the
underlying data licence itself is free and unrestricted. Worth planning around if the goal is bulk vector
pull rather than WMS tiles.
