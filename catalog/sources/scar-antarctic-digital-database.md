---
id: scar-antarctic-digital-database
name: SCAR Antarctic Digital Database (ADD)
operator: "[[British Antarctic Survey]]"
portal: https://add.scar.org/
scale_tier: continental
resolution: "high-resolution (1:250,000-class) and medium-resolution (1:1,000,000-class) vector themes"
data_types: [geologic-map]
regions: [antarctica]
coverage:
  name: Antarctica (south of 60°S)
  bbox: [-180.0, -90.0, 180.0, -60.0]
fetch:
  - method: http-download
    url: https://data.bas.ac.uk/items/e74543c0-4c4e-4b41-aa33-5bb2f67df389/
    notes: >-
      Eight vector themes at two resolutions: coastline (ice/rock/grounding-line/ice-shelf
      front), rock outcrop (Landsat-8-derived + manual), 100/500 m contours, lakes, moraine,
      streams (Byers Peninsula, Transantarctic Mtns only — incomplete), seamask, and the
      60°S data-limit line. Viewer at add.scar.org links out to this BAS Data Catalogue item.
formats: [shp, gpkg]
license: CC BY-ND 4.0 (SCAR/BAS)
added: 2026-08-11
verified: 2026-08-11
---

The reference topographic/coastline vector basemap for Antarctica, maintained by BAS on
behalf of SCAR. Rock outcrop extent (ice-free bedrock vs. ice) is the closest thing to a
geology signal here — there is no fault/fold or bedrock-unit layer; for that, pair with the
GeoMAP entry. The No-Derivatives clause on the CC BY-ND license is a real constraint:
redistributing a modified/reprojected extract needs a case-by-case check against that term
rather than assuming standard CC-BY reuse.
