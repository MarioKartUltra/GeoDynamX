---
id: emag2v3-magnetic-grid
name: EMAG2v3 — Earth Magnetic Anomaly Grid (2-arc-min)
operator: "[[NOAA NCEI]]"
portal: https://www.ncei.noaa.gov/products/earth-magnetic-model-anomaly-grid-2
scale_tier: global
resolution: 2 arc-min grid (upward-continued to 4 km, and sea-level compilation for ocean/coastal areas)
data_types: [magnetics]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530.zip
    notes: >-
      4.5 GB zipped ASCII/CSV grid (lon, lat, upward-continued anomaly, sea-level
      anomaly, code). Landing page also links GeoTIFF, PNG and KMZ renderings and
      an Esri ArcGIS Hub image service.
  - method: http-download
    url: https://www.ngdc.noaa.gov/geomag/emag2_download.html
    notes: primary download index page (all format links live from here)
formats: [csv, geotiff, kmz, png]
native_metadata:
  - format: iso-19115-xml
    url: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.geophysical_models%3AEMAG2_V3
license: public domain (US federal); cite via DOI 10.7289/V5H70CVX
added: 2026-08-11
verified: 2026-08-11
---

Global magnetic anomaly grid compiled from satellite, marine and airborne track-line data
(11.5M+ new points over EMAG2). v3 drops interpolation into unmeasured areas so gaps in
survey coverage are visible rather than smoothed over — worth knowing before treating this
as a uniform-density input. No auth wall; direct zip download.
