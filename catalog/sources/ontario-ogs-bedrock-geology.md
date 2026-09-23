---
id: ontario-ogs-bedrock-geology
name: Ontario Geological Survey — 1:250,000 Bedrock Geology of Ontario
operator: "[[Ontario Geological Survey]]"
portal: https://data.ontario.ca/dataset/1250-000-scale-bedrock-geology-of-ontario
scale_tier: regional
resolution: "1:250,000"
data_types: [structural-vectors, geologic-map]
regions: [north-america, canada, ontario]
coverage:
  name: Ontario
  bbox: [-95.2, 41.7, -74.3, 56.9]
fetch:
  - method: manual
    url: https://data.ontario.ca/dataset/1250-000-scale-bedrock-geology-of-ontario
    notes: ESRI shapefile resource resolves through the GeologyOntario persistent-linking viewer (publication MRD126-REV1), no static zip URL exposed
  - method: http-download
    url: https://www.geologyontario.mndm.gov.on.ca/mines/data/google/mrd126/doc.kml
    notes: same compilation as KML, direct download, verified live
license: Ministry Terms of Use (open, attribution + Crown copyright notice required; see geologyontario.mndm.gov.on.ca/terms_of_use.html)
added: 2026-08-11
verified: 2026-08-11
---

A genuine seamless provincial compilation, not just an index: bedrock units, major faults, dike
swarms, iron formations, and kimberlites at 1:250,000, part of the older "OGSEarth" product line.
Provider is inconsistent about license naming (Ontario open-data listings say "Ministry Terms of Use"
rather than the province's usual Open Government Licence), so quote attribution requirements from the
terms-of-use page rather than assuming blanket OGL terms. The shapefile itself sits behind a legacy
JS viewer (persistent-linking) with no scrapable direct URL; the KML mirror is the one link that is
directly fetchable.
