---
id: texas-beg-statemap
name: Texas BEG — STATEMAP Geologic Quadrangle GIS Data
operator: "[[Bureau of Economic Geology]]"
portal: https://www.beg.utexas.edu/research/areas/geologic-mapping
scale_tier: regional
resolution: "1:24,000 (STATEMAP quadrangles)"
data_types: [structural-vectors, geologic-map]
regions: [north-america, united-states, texas]
coverage:
  name: Texas
  bbox: [-106.6, 25.8, -93.5, 36.5]
fetch:
  - method: manual
    url: https://www.beg.utexas.edu/research/areas/geologic-mapping
    notes: per-quadrangle shapefile zips (Austin, Del Rio, Georgetown, New Braunfels, Galveston Island), most along the Balcones Fault Zone
  - method: http-download
    url: https://www.beg.utexas.edu/files/content/beg/1956/Austin.zip
    notes: verified example quadrangle download, ArcView shapefile format
formats: [shp]
license: not explicitly stated; free download from BEG
added: 2026-08-11
verified: 2026-08-11
---

BEG's STATEMAP output is quadrangle-scale, not statewide-seamless: each zip is one 1:24,000 sheet,
several of them (Austin, New Braunfels, Georgetown, Del Rio) sitting directly on the Balcones Fault
Zone and carrying fault-trace linework. The older statewide product one might expect —
the digitized 38-sheet Geologic Atlas of Texas / Geologic Database of Texas (GDT), long hosted by
TNRIS — could not be verified: both `tnris.org` and `feature.tnris.org` fail to resolve (NXDOMAIN)
following the 2023 TNRIS→Texas Geographic Information Office (TxGIO) rename, and no live replacement
endpoint was found. Treat statewide-seamless Texas coverage as an open gap for now.
