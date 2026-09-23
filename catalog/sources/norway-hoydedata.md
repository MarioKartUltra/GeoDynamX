---
id: norway-hoydedata
name: Høydedata / Nasjonal detaljert høydemodell — Norway national LiDAR elevation
operator: "[[Kartverket]]"
portal: https://hoydedata.no/LaserInnsyn2/
scale_tier: national
resolution: "1 m DTM/DSM grid, seamless nationwide (completed 2022); source LiDAR point cloud up to 10 pts/m2, classified (ground/vegetation/building/water)"
data_types: [dem, lidar]
regions: [europe, norway]
coverage:
  name: Norway (mainland; excludes Svalbard)
  bbox: [4.5, 57.9, 31.5, 71.2]
fetch:
  - method: manual
    url: https://hoydedata.no/LaserInnsyn2/
    notes: "LaserInnsyn 2 viewer — confirmed live; AOI-based browse/order for point cloud and DTM/DSM tiles"
  - method: wcs
    url: https://wms.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833?service=WCS&request=GetCapabilities
    notes: "Geonorge WCS for the national DTM (Nasjonal høydemodell), EPSG:25833 — GetCapabilities confirmed live"
  - method: api
    url: https://hoydedata.no/LaserServices/rest/DownloadFile.ashx
    notes: "confirmed live; REST endpoint used by LaserInnsyn's own AOI point-cloud export"
formats: [geotiff, laz]
license: "NLOD 2.0 (Norwegian Licence for Open Government Data — CC BY-equivalent, attribution to Kartverket)"
added: 2026-08-11
verified: 2026-08-11
---

Kartverket's Nasjonal detaljert høydemodell (NDH) programme flew ~230,000 km² of airborne LiDAR
between 2014 and 2022 to complete a seamless 1 m DTM/DSM for the entire mainland, with the raw
classified point cloud (up to 10 pts/m2) published alongside. Everything funnels through
høydedata.no's LaserInnsyn viewer for AOI-based ordering; the underlying Geonorge WCS gives
programmatic raster access without going through the viewer. Svalbard is mapped separately and
isn't part of this seamless national layer.
