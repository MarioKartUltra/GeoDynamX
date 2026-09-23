---
id: usgs-world-geologic-provinces
name: USGS Geologic Provinces of the World (2000 World Petroleum Assessment)
operator: "[[U.S. Geological Survey]]"
portal: https://www.sciencebase.gov/catalog/item/60ad2fa1d34e4043c850ed98
scale_tier: global
resolution: "reconnaissance-scale province polygons; no fixed map scale (compiled for the 2000 World Petroleum Assessment, roughly 1:5,000,000-class generalization)"
data_types: [geologic-map]
regions: [global]
coverage:
  name: Global (assessed petroleum geologic provinces)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://www.sciencebase.gov/catalog/file/get/60ad2fa1d34e4043c850ed98
    notes: >-
      Downloads GeologicProvinc.zip → WEP_PRVA shapefile (arcs + polygons) plus FGDC XML
      metadata. Assessed provinces only (Priority + Boutique); no auth.
formats: [shp]
license: public domain (US federal)
added: 2026-08-11
verified: 2026-08-11
---

Province-level geologic-map generalization used for the USGS World Petroleum Assessment
2000 (DDS-60): boundaries encode dominant lithology, stratigraphic age and structural
style rather than a bedrock-unit map. Coarse by design — this is a global reconnaissance
compilation, not a substitute for a national geologic map. Offshore boundaries mostly
follow the 2000 m (occasionally 4000 m) bathymetric contour rather than a geologic
criterion, which shows up as a straight-line artifact near continental margins.
