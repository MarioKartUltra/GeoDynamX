---
id: gebco-2026-grid
name: GEBCO_2026 Grid
operator: "[[General Bathymetric Chart of the Oceans]]"
portal: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
scale_tier: global
resolution: "15 arc-second grid (~450 m at equator)"
data_types: [bathymetry, dem]
regions: [global]
coverage:
  name: Global (land + ocean)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/geotiff/gebco_2026_geotiff.zip?download=1
    notes: >-
      Full global grid, ice-surface version, GeoTIFF (multi-GB zip); netCDF and Esri
      ASCII siblings, plus a sub-ice topography/bathymetry version and a Type Identifier
      (TID) grid, at parallel paths under the same tree. No login; verified live 2026-08-11.
      The site's own "user-selected area" tool (download.gebco.net) does not currently
      resolve.
  - method: api
    url: https://data.ceda.ac.uk/bodc/gebco/global/gebco_2026
    notes: OPeNDAP endpoint for subsetting without pulling the whole grid.
formats: [netcdf, geotiff, esri-ascii]
license: Public domain (attribution requested)
added: 2026-08-11
verified: 2026-08-11
native_metadata:
  - format: DOI landing page
    url: https://doi.org/10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa
---

The standard global bathymetry+topography grid, compiled under the joint IOC-UNESCO/IHO
GEBCO program and mirrored via BODC/CEDA. GEBCO_2026 (Apr 2026) ships as global files
(multi-GB compressed) plus OPeNDAP subsetting; the advertised interactive area-select
tool at download.gebco.net did not resolve at verification time. The companion TID grid
flags measured soundings vs. interpolated/satellite-derived cells — check it before
treating GEBCO depths as survey-grade in undersampled ocean basins.
