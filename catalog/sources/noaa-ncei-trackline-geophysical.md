---
id: noaa-ncei-trackline-geophysical
name: NOAA NCEI Marine Trackline Geophysical Data
operator: "[[NOAA NCEI]]"
portal: https://www.ncei.noaa.gov/products/marine-trackline-geophysical-data
scale_tier: global
resolution: "single-beam soundings and single/analog-channel seismic along ship tracks, 1939-present; spacing set by cruise track spacing, not a grid"
data_types: [bathymetry, seismic-reflection]
regions: [global]
coverage:
  name: Global ship-track coverage
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://www.ncei.noaa.gov/maps/trackline-geophysics/
    notes: >-
      Interactive Trackline Geophysical Data Viewer (verified live 2026-08-11): filter by
      location/year/institution/platform/data-type, then export per-cruise.
formats: [mgd77t, segy, tiff]
license: Public domain (US federal)
added: 2026-08-11
verified: 2026-08-11
---

Pre-multibeam-era (and still-ongoing) trackline archive: single-beam bathymetry, subbottom
profiles, magnetics, gravity, side-scan, and historic single/analog-channel seismic
reflection/refraction along ship tracks back to 1939. Digital surveys export as MGD77T
(navigation+geophysics); ancillary high-volume data (seismic, sonar) comes as SEG-Y or
similar industry formats, and older analog seismic sections are scanned TIFF images only —
not digitized SEG-Y — so "seismic reflection" here can mean a scan of a paper record, not a
trace file.
