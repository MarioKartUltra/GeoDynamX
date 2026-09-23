---
id: nasa-earthdata-cmr
name: "NASA Earthdata Common Metadata Repository (CMR)"
operator: "[[NASA]]"
portal: https://search.earthdata.nasa.gov/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [dem, lidar, gravity]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://cmr.earthdata.nasa.gov/search/collections.json?keyword=landsat&page_size=1
    notes: >-
      Confirmed live: CMR collection-search returns real JSON metadata (dataset_id,
      concept_id, boxes, time_start, etc.). Same endpoint family covers granule search
      at /search/granules.json. This is NASA's single metadata catalog fronting all
      12 Earth-science DAACs (LP DAAC, PO.DAAC, NSIDC DAAC, ASF DAAC, etc.).
  - method: manual
    url: https://search.earthdata.nasa.gov/
    notes: "Earthdata Search — the point-and-click UI over the same CMR catalog; confirmed live (HTTP 200)."
formats: [geotiff, hdf5, netcdf, csv, las]
license: "Mostly open/public-domain NASA data; some contributed/partner collections carry their own terms — check per-collection metadata"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** CMR is the single metadata-search API (and Earthdata Search the UI on top
of it) that fronts essentially all of NASA's Earth-science holdings across every DAAC — one
collection/granule search interface standing in front of thousands of distinct datasets. Free
Earthdata Login is required for actual data download (not for search). Flagship
geology-relevant holdings reachable through it: NASADEM / SRTM / ASTER GDEM elevation
(LP DAAC — `dem`), GEDI full-waveform lidar footprints (ORNL/LP DAAC — `lidar`), GRACE/GRACE-FO
gravity-field products (PO.DAAC — `gravity`), plus ICESat-2 laser altimetry and the full
Landsat/ASTER/EMIT optical-imagery archive. Harmony
(confirmed live at https://harmony.earthdata.nasa.gov/, HTTP 200) sits alongside CMR as an
on-the-fly subsetting/reformatting service for a growing subset of these same collections —
useful for clipping a global product to an AOI before download rather than pulling a whole
granule.
