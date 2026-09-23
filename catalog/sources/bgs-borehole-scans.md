---
id: bgs-borehole-scans
name: BGS Single Onshore Borehole Index (SOBI) and Scanned Borehole Records
operator: "[[British Geological Survey]]"
portal: https://www.bgs.ac.uk/information-hub/borehole-records/
scale_tier: sample
resolution: "point-located borehole/core records with scanned logs, depth-registered lithological descriptions"
data_types: [borehole, core-sample]
regions: [europe, united-kingdom]
coverage:
  name: Great Britain (onshore)
  bbox: [-8.649, 49.823, 1.763, 60.845]
fetch:
  - method: api
    url: https://ogcapi.bgs.ac.uk/collections/onshoreboreholeindex
    notes: OGC API-Features endpoint for the SOBI point dataset
  - method: http-download
    url: https://www.bgs.ac.uk/download/single-onshore-borehole-index-sobi-dataset/
    notes: bulk SOBI download (GIS point data, ESRI/MapInfo; other formats on request)
  - method: manual
    url: https://mapapps2.bgs.ac.uk/geoindex/home.html?layer=BGSBoreholes
    notes: GeoIndex viewer — click-through to scanned borehole/core log images (PDF)
license: Open Government Licence
added: 2026-08-11
verified: 2026-08-11
---

National Geoscience Data Centre's index of over a million onshore Great Britain boreholes,
shafts and wells, each linking to a scanned record (geologist's lithological log, depth/
thickness, sometimes geophysical logs). SOBI itself is just point locations + metadata via OGC
API-Features or bulk download; the actual scanned logs are viewed one at a time through the
GeoIndex map, not bulk-downloadable. Physical core material referenced in these records is
held separately at the BGS National Geological Repository (Keyworth).
