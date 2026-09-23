---
id: boem-gom-bathymetry
name: BOEM Northern Gulf of Mexico Deepwater Bathymetry Grid
operator: "[[Bureau of Ocean Energy Management]]"
portal: https://www.boem.gov/oil-gas-energy/mapping-and-data
scale_tier: regional
resolution: 12.2 m posting (resolves ~25-50 m; seismic water-bottom pick, not multibeam)
data_types: [bathymetry]
regions: [north-america, gulf-of-mexico]
coverage:
  name: Northern Gulf of Mexico deepwater
  bbox: [-97.0, 26.0, -86.0, 30.0]
fetch:
  - method: http-download
    url: https://www.boem.gov/oil-gas-energy/mapping-and-data/map-gallery/northern-goa-deepwater-bathymetry-grid-3d-seismic
    notes: >-
      Product page hosts GeoTIFF grids as East/West-split zips, in feet or meters
      (e.g. BOEM_Bathymetry_East_meters_tiff.zip, ~490 MB) plus hillshade and contour
      zips. The generic mapping-and-data URL is live but is the general portal,
      not this product page; the older boem-northern-gulf-mexico... map-gallery slug
      301-redirects here under a "GoA" (Gulf of America) slug.
formats: [geotiff]
license: public domain (US federal)
added: 2026-08-11
verified: 2026-08-11
---

The ML fault-mapping project's dataset. NOT multibeam: seismic water-bottom pick
mosaicked from ~100 3-D surveys (Kramer & Shedd 2017) — usable scaling range ≈ 1-2.5
octaves, per-survey knee required. Mosaic seams are fault-shaped; see the research note.
