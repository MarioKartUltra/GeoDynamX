---
id: ga-surface-geology-faults
name: Geoscience Australia Surface Geology of Australia — Faults & Structure
operator: "[[Geoscience Australia]]"
portal: https://digital.atlas.gov.au/datasets/faults-surface-geology-11-million-scale/about
scale_tier: national
resolution: "1:1,000,000 (2012 edition); 1:2,500,000 edition also available"
data_types: [structural-vectors, geologic-map]
regions: [australia]
coverage:
  name: Australia
  bbox: [114.2288, -54.7732, 158.9361, -10.6409]
fetch:
  - method: arcgis-rest
    url: https://services.ga.gov.au/gis/rest/services/GA_Surface_Geology/MapServer/7
    notes: >-
      Faults layer (brittle/ductile structures, movement sense, dip, age) —
      sibling layers in the same MapServer carry geology units and contacts.
      Confirmed live and public via the ArcGIS item record (owner aus_digitalatlas).
  - method: http-download
    url: https://data.gov.au/data/dataset/surface-geology-of-australia-1-1-million-scale-dataset-2012-edition
    notes: packaged shapefile/file-geodatabase download of the full seamless 1:1M compilation
formats: [shp, fgdb, esri-feature-service]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

The national seamless bedrock compilation (first released 2008, current edition 2012):
outcrop geology, regolith, contacts, and — the layer of interest here — faults/shears with
type, movement sense, dip and age, GeoSciML-aligned attributes. Distinct from GSWA's WA-only
product already in the catalog; this one edge-matches the whole continent, with a coarser
1:2.5M edition alongside it. Served as an Esri feature service under Digital Atlas of
Australia branding, but GA remains the data owner and metadata contact.
