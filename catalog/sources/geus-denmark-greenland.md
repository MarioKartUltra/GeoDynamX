---
id: geus-denmark-greenland
name: GEUS Map Database — Denmark and Greenland geology, gravity, magnetics
operator: "[[Geological Survey of Denmark and Greenland]]"
portal: https://eng.geus.dk/products-services-facilities/data-and-maps
scale_tier: national
resolution: "1:25,000-1:200,000 (Denmark); 1:100,000-1:2,500,000 (Greenland)"
data_types: [structural-vectors, geologic-map, gravity, magnetics]
regions: [europe, denmark, greenland]
coverage:
  name: Denmark and Greenland
  bbox: [-73.0, 54.5, 15.5, 84.0]
fetch:
  - method: arcgis-rest
    url: https://data.geus.dk/arcgis/rest/services/Denmark/Strukturelle_elementer/MapServer?f=json
    notes: "Denmark structural elements (faults/basin-margin lines) — confirmed live"
  - method: arcgis-rest
    url: https://data.geus.dk/arcgis/rest/services/Denmark/Jordartskort_200000/MapServer?f=json
    notes: "Denmark surficial/bedrock geology 1:200,000 — confirmed live"
  - method: arcgis-rest
    url: https://data.geus.dk/arcgis/rest/services/Greenland/Geological_map_500k/MapServer?f=json
    notes: "seamless Greenland geological map 1:500,000 — confirmed live"
  - method: arcgis-rest
    url: https://data.geus.dk/arcgis/rest/services/Denmark/Gravimetri/MapServer?f=json
    notes: "Denmark gravity survey — confirmed live"
  - method: arcgis-rest
    url: https://data.geus.dk/arcgis/rest/services/Greenland/Magnetic_compilation/MapServer?f=json
    notes: "Greenland aeromagnetic compilation — confirmed live; service root https://data.geus.dk/arcgis/rest/services lists Denmark/DKModel2019/Europe/Greenland/Havvind folders"
  - method: http-download
    url: https://dataverse.geus.dk/dataverse/geological_maps_greenland
    notes: "bulk geological-map downloads (ArcGIS .mpkx, shapefile) with per-dataset DOIs, 150+ Greenland map sheets"
license: "Free reuse (GEUS open-data policy; Danish deep-subsurface data made freely accessible 2022); individual Dataverse datasets carry their own CC0/CC-BY-style terms"
added: 2026-08-11
verified: 2026-08-11
---

One national survey, two very different domains: Denmark's onshore/offshore geology (structural elements,
Jordartskort surficial+bedrock maps, gravimetry) and Greenland's seamless bedrock geology plus decades of
airborne magnetic/radiometric surveys. All served from the same ArcGIS REST catalogue (data.geus.dk),
confirmed live with real layer listings — genuinely covers structural vectors, geologic maps, and geophysics
in a single operator. Greenland map sheets are also archived with DOIs on GEUS's Dataverse for bulk pull.
No registration wall found for browsing/downloading; some deep-subsurface Denmark datasets were paywalled
until GEUS opened them in 2022.
