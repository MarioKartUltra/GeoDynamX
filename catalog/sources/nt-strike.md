---
id: nt-strike
name: NT STRIKE — Northern Territory Geoscience Data System
operator: "[[Northern Territory Geological Survey]]"
portal: https://strike.nt.gov.au/wss.html
scale_tier: regional
resolution: "1:2,500,000 statewide geology/faults compilation; local mapping to 1:100,000-1:250,000"
data_types: [structural-vectors, geologic-map, borehole]
regions: [australia, northern-territory]
coverage:
  name: Northern Territory
  bbox: [128.9, -26.1, 138.1, -10.9]
fetch:
  - method: http-download
    url: https://geoscience.nt.gov.au/downloads/NTWideDownloads.html
    notes: >-
      Direct NT-wide zips: NT_Geology_shp.zip, GEO_FAULTS_2500K_shp.zip,
      GEO_INTERP_2500K_shp.zip, GEOLOGY_PROVINCES_2.5M_shp.zip, plus
      drilling/geochem/geophysics/titles. Links are wrapped through a
      strike.nt.gov.au/link/ redirector but land on real geoscience.nt.gov.au
      zips — a fault-layer zip was confirmed downloadable (HTTP 200, real
      Content-Length).
  - method: manual
    url: https://strike.nt.gov.au/wss.html
    notes: interactive viewer for AOI query/download of the same layers
formats: [shp, tab, kml, csv]
license: CC BY
added: 2026-08-11
verified: 2026-08-11
---

STRIKE is NT's map viewer; the NT-Wide Downloads page behind it is the friction-free path —
statewide zipped shapefiles for geology, faults (GEO_FAULTS_2500K), interpreted geology,
provinces, drillholes, geochemistry and titles, no login. Files route through a
strike.nt.gov.au redirect wrapper but resolve to real geoscience.nt.gov.au zips. Detailed
local mapping beyond the statewide 1:2.5M compilation lives in NTGS's separate GEMIS
report/data search, not covered here.
