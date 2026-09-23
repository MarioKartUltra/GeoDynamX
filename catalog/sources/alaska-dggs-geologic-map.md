---
id: alaska-dggs-geologic-map
name: Alaska DGGS — Geologic Map Index & Statewide Geologic Map of Alaska
operator: "[[Alaska Division of Geological & Geophysical Surveys]]"
portal: https://dggs.alaska.gov/maps-data/index.html
scale_tier: regional
resolution: "1:500,000 (detailed compilation); 1:1,584,000 (generalized); index covers 1:24,000-1:250,000 source sheets"
data_types: [structural-vectors, geologic-map]
regions: [north-america, united-states, alaska]
coverage:
  name: Alaska
  bbox: [-179.15, 51.2, -129.9, 71.44]
fetch:
  - method: manual
    url: https://maps.dggs.alaska.gov/mapindex/
    notes: DGGS/USGS Geologic Map Index of Alaska — searchable footprints linking out to individual publications
  - method: http-download
    url: https://pubs.usgs.gov/sim/3340/sim3340_shp.zip
    notes: "USGS SIM 3340 (Wilson & others, 2015) statewide seamless geodatabase, shapefile export, ~710MB; feature classes include AKStategeolarc_generalized (major faults shown on published map) and AKStategeol_dike / _lineament"
formats: [gdb, shp]
license: Public Domain (U.S. Government Work) for SIM 3340; individual DGGS-authored publications otherwise under State of Alaska terms
added: 2026-08-11
verified: 2026-08-11
---

DGGS itself is primarily a discovery layer — the Geologic Map Index catalogs footprints of DGGS and
USGS map sheets and links out to per-publication downloads, most of which are scanned or quadrangle-
scale. The one statewide seamless vector product, SIM 3340, is a USGS publication (compiled with DGGS
data) rather than a DGGS release, but it is the map DGGS's own index points to for statewide coverage,
and its readme confirms a dedicated major-faults line feature class alongside the geology polygons —
a genuine structural-vectors + geologic-map pairing for this state.
