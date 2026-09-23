---
id: noaa-ncei-arcgis-gateway
name: "NOAA NCEI/NGDC ArcGIS REST Services Directory"
operator: "[[NOAA NCEI]]"
portal: https://www.ncei.noaa.gov/maps/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [bathymetry, magnetics, dem]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: arcgis-rest
    url: https://gis.ngdc.noaa.gov/arcgis/rest/services?f=json
    notes: >-
      Confirmed live: services-directory root returns real JSON listing dozens of MapServer/
      ImageServer endpoints across folders (DEM_mosaics, multibeam_mosaics, antarctic,
      arctic_ps, IHO, NOS_MBAB, etc.), including bag_bathymetry / multibeam_mosaic
      (`bathymetry`), EMAG2v3 magnetic-anomaly ImageServer/MapServer (`magnetics`), and
      etopo1 plus the DEM_mosaics folder (`dem`). This is the actual machine-readable
      catalog-of-services behind NCEI's marine-geophysical holdings; NOAA's older
      CKAN-based data.noaa.gov API and NGDC's legacy CSW geoportal are both decommissioned
      (confirmed 404 on all tested paths, 2026-08-11) — this ArcGIS REST directory is what is
      actually live today.
  - method: manual
    url: https://data.noaa.gov/onestop
    notes: >-
      NOAA's current catalog-browse UI (successor branding to "OneStop"); confirmed live
      (HTTP 200), though its own search backend (onestop-search) returned 404 on every path
      tested and appears retired — treat this UI as a portal, not as evidence of a live
      OneStop API.
formats: [geotiff, esri-grid, png]
license: "Public domain (U.S. Government work)"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** Rather than a single documented "OneStop search API" (the branded OneStop
backend appears retired — every guessed endpoint 404'd on live testing), the API that
actually fronts many of NCEI's geophysical grids and map layers today is the standard ArcGIS
REST Services Directory at `gis.ngdc.noaa.gov`: one `?f=json` root call enumerates dozens of
independently-browsable MapServer/ImageServer services, each queryable/exportable through the
standard ArcGIS REST protocol. Flagship geology-relevant holdings reachable through it: BAG-
format multibeam bathymetry mosaics and hillshades (`bathymetry`), EMAG2v3 Earth Magnetic
Anomaly Grid (`magnetics`), ETOPO global relief and other DEM mosaics (`dem`), plus
region-specific bathymetry (Gulf Data Atlas, Antarctic/Arctic polar stereographic layers) and
tsunami/hazard model layers. A separate, unrelated NCEI Search Service
(`ncei.noaa.gov/access/services/search/v1`, confirmed live) also exists but fronts climate/
weather station records (GHCN, ISD, climate normals) — verified during this pass to carry no
data types in this catalog's geology-oriented enum, so it is intentionally not the entry
recorded here.
