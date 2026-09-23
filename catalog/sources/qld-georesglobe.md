---
id: qld-georesglobe
name: Queensland GeoResGlobe / Geoscientific Information Services
operator: "[[Geological Survey of Queensland]]"
portal: https://georesglobe.information.qld.gov.au/
scale_tier: regional
resolution: "basin/province scale (StructuralFramework) to ~1:100,000 (GeologyDetailed)"
data_types: [structural-vectors, geologic-map, borehole]
regions: [australia, queensland]
coverage:
  name: Queensland
  bbox: [137.9, -29.2, 153.6, -9.1]
fetch:
  - method: arcgis-rest
    url: https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation/StructuralFramework/MapServer
    notes: >-
      Basins/provinces/orogens structural framework (100+ named layers). Sibling
      services in the same GeoscientificInformation folder (GeologyDetailed,
      GeologyRegional, GeologyObservations, Boreholes) carry map units and point
      data; all confirmed reachable without login.
  - method: manual
    url: https://georesglobe.information.qld.gov.au/
    notes: interactive viewer; splash screen offers several login providers but most layers are public underneath
formats: [shp, tab, fgdb, kml, gpkg]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

Queensland's geoscience layers live as a plain ArcGIS Server folder
(GeoscientificInformation) with named services for structural framework, detailed/regional
geology, observation points and boreholes — reachable directly, no login needed, and
license-confirmed CC BY 4.0 via the data.qld.gov.au CKAN catalogue. GeoResGlobe is the
branded viewer in front of the same services; its login screen is optional for browsing.
