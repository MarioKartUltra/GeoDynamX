---
id: cgs-south-africa-geoscience-webmap
name: Council for Geoscience Interactive Web Map (South Africa)
operator: "[[Council for Geoscience]]"
portal: https://maps.geoscience.org.za/
scale_tier: national
resolution: "variable; national map series at 1:250,000 and 1:1,000,000, geophysics grids at survey resolution"
data_types: [geologic-map, structural-vectors, gravity, magnetics]
regions: [africa, south-africa]
coverage:
  name: South Africa
  bbox: [16.3, -34.9, 32.9, -22.1]
fetch:
  - method: arcgis-rest
    url: https://maps.geoscience.org.za/server/rest/services?f=json
    notes: "confirmed live ArcGIS Enterprise 11.5; public folders incl. 8_GEOPHYSICS_DATA, 5_SEISMICITY, ENGINEERING_GEOLOGY — some individual services return 'Token Required'"
  - method: manual
    url: https://www.geoscience.org.za/cgs/systems/publications/downloadable-material/
    notes: "downloadable map/document library (Simplified Geological Map of South Africa, etc.)"
formats: [shp, gdb, tiff]
license: "Geoscience Act 100 of 1993 (as amended) — CGS data made available to stakeholders; some ArcGIS layers gated behind a token"
added: 2026-08-11
verified: 2026-08-11
---

The Council for Geoscience (successor to the Geological Survey of South Africa, 1912) runs an ArcGIS
Enterprise deployment with public folders covering geophysics, seismicity, and engineering geology
alongside its national geological map series. The service root and folder listing are open; several
individual layers demand an ArcGIS token, so plan on a registration step for anything beyond the
published downloadable maps/documents.
