---
id: usgs-sciencebase
name: "USGS ScienceBase Catalog API"
operator: "[[U.S. Geological Survey]]"
portal: https://www.sciencebase.gov/catalog/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [geologic-map, borehole, core-sample, structural-vectors, outcrop-model]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://www.sciencebase.gov/catalog/items?format=json&q=basalt&max=1
    notes: >-
      Confirmed live: full-text item search returns real JSON hits (505 items for "basalt",
      1374 for "fault", 53 for "core sample", 5 for "structural geology shapefile", 1 for
      "outcrop model" — spot-checked 2026-08-11). No auth needed for public-item search;
      no API key required.
  - method: manual
    url: https://www.sciencebase.gov/catalog/
    notes: "ScienceBase catalog browse UI — confirmed live (HTTP 200)."
formats: [shapefile, geotiff, csv, pdf, zip]
license: "Mostly public domain (U.S. Government work); some contributed/partner items carry their own terms — check per-item metadata"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** ScienceBase is USGS's general-purpose data/document repository — one REST
search API fronting hundreds of thousands of individual "items" spanning nearly every USGS
science mission area, not a single discipline. Spot-checked flagship geology-relevant
holdings reachable through it: state and quadrangle geologic maps (`geologic-map`), borehole
lithology/erosion-test datasets (`borehole`), oil/gas and mineral-exploration core-sample
photo/description sets (`core-sample`), fault and structural-geology shapefiles including
Quaternary-fault slip/dilation-tendency calculations (`structural-vectors`), and
structure-from-motion outcrop models (`outcrop-model`). Because it is a general repository
rather than a curated discipline catalog, search relevance and metadata completeness vary
item-to-item — this is the right place to *discover* an obscure USGS dataset, not necessarily
the fastest path to a specific, well-known one (those usually have their own catalog entry,
e.g. the 3DEP, NGMDB, or SGMC entries in this catalog).
