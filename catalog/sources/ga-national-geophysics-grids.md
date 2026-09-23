---
id: ga-national-geophysics-grids
name: Geoscience Australia National Geophysical Grids — Gravity & Magnetics
operator: "[[Geoscience Australia]]"
portal: https://portal.ga.gov.au/persona/gadds
scale_tier: national
resolution: "magnetics ~88 m cell (7th ed., 2019/2020); gravity 400 m cell, station spacing ~1-11 km"
data_types: [gravity, magnetics]
regions: [australia]
coverage:
  name: Australia (onshore + continental margin)
  bbox: [108.0, -48.0, 164.0, -8.0]
fetch:
  - method: wcs
    url: https://services.ga.gov.au/gis/geophysical-grids/ows
    notes: OWS endpoint (WMS/WCS) serving national gravity, magnetic, radiometric and elevation grids for QGIS/ArcGIS
  - method: http-download
    url: https://thredds.nci.org.au/thredds/catalog/iv65/Geoscience_Australia_Geophysics_Reference_Data_Collection/national_geophysical_compilations/catalog.html
    notes: >-
      NCI THREDDS catalogue — direct NetCDF/GeoTIFF grid files (e.g. Magmap2019
      pseudogravity/RTP grids ~88 m, National Gravity Compilation 2019 400 m),
      plus NCSS and OPeNDAP subsetting. Sample file confirmed live (2.65 GB .nc,
      HTTP 200).
  - method: manual
    url: https://portal.ga.gov.au/persona/gadds
    auth: free-registration
    notes: GADDS2 — survey-level search and bulk order across 4000+ individual geophysical surveys
formats: [netcdf, geotiff, ers]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

National-scale gravity (400 m cell, 2019 compilation) and magnetic (7th-edition, ~88 m cell)
grids merging ground, airborne, marine and satellite data into seamless continental
coverage. Direct NetCDF/GeoTIFF pulls work straight off NCI's THREDDS server — verified a
2.65 GB magnetics file resolves with a real Content-Length. The geophysical-grids OWS
endpoint serves the same compilations as WMS/WCS for desktop GIS; GADDS2 is the path for
individual survey-level data rather than the merged grids.
