---
id: aster-gdem-v3
name: ASTER Global Digital Elevation Model V3 (ASTGTM)
operator: "[[National Aeronautics and Space Administration]]"
portal: https://www.earthdata.nasa.gov/data/catalog/lpcloud-astgtm-003
scale_tier: global
resolution: "30 m posting (1 arc-second)"
data_types: [dem]
regions: [global]
coverage:
  name: Global land, 83°S-83°N
  bbox: [-180.0, -83.0, 180.0, 83.0]
fetch:
  - method: manual
    url: https://search.earthdata.nasa.gov/search?q=ASTGTM
    auth: free-registration
    notes: >-
      NASA Earthdata login required. Collection ASTGTM_003 (DOI 10.5067/ASTER/ASTGTM.003),
      1x1 degree Cloud-Optimized GeoTIFF tiles via LP DAAC.
  - method: api
    url: https://cmr.earthdata.nasa.gov/search/granules.json?concept_id=C1711961296-LPCLOUD
    notes: >-
      CMR granule-search API for the collection; verified live JSON response (2026-08-11).
      Metadata access needs no auth, but granule download links do.
formats: [cog, netcdf4]
license: No-cost, no restriction (NASA/METI open data policy; joint US/Japan product)
native_metadata:
  - format: json (CMR/ECHO collection record)
    url: https://cmr.earthdata.nasa.gov/search/collections.json?concept_id=C1711961296-LPCLOUD
added: 2026-08-11
verified: 2026-08-11
---

Stereo-derived from ASTER imagery (2000-2013), a joint NASA/Japan METI product
distributed through LP DAAC. V3 (2019) improved stereo-pair coverage and accuracy over
V2. Covers ~99% of Earth's land between 83°N/S as 1°x1° tiles with a DEM band plus a
per-pixel scene-count band (NUM) for quality screening. Free but Earthdata-login-gated;
known for more speckle/artifacts than SRTM or Copernicus DEM in low-relief terrain.
