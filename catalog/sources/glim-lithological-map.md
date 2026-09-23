---
id: glim-lithological-map
name: GLiM — Global Lithological Map database v1.0
operator: "[[University of Hamburg]]"
portal: https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html
scale_tier: global
resolution: "~1:3,750,000 average source-map scale (vector, 1.24M polygons); also released gridded at 0.5 degrees"
data_types: [geologic-map]
regions: [global]
coverage:
  name: Global (emerged land surface)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: "https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1"
    notes: >-
      Vector geodatabase (LiMW_GIS 2015.gdb.zip), 3-level lithological classification,
      ~1.24M polygons from 92 regional source maps (Hartmann & Moosdorf, 2012). Link is a
      personal Dropbox mirror embedded on the official Univ. Hamburg page, not an
      institutional host — treat as fetch-and-cache, it can move without notice. A
      gridded 0.5-degree raster version plus documentation is archived at PANGAEA/AWI
      EPIC (hdl:10013/epic.39939), no shapefile there.
formats: [gdb]
license: Free for scientific use (Hartmann & Moosdorf, 2012); no explicit open-data licence stated on the source page
added: 2026-08-11
verified: 2026-08-11
---

The standard global lithology (not stratigraphy) map: rock-type polygons rather than
formation/age units, built for weathering and geochemical-flux modeling — useful here as
a coarse global geology backdrop where no finer national map exists. ~100x more detailed
than prior global lithological compilations, but still averaging a 1:3.75M source scale,
so treat it as a basemap, not a structural dataset — it carries no fault/fold vectors.
