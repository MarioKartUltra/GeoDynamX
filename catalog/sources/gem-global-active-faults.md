---
id: gem-global-active-faults
name: GEM Global Active Faults Database (GAF)
operator: "[[GEM Foundation]]"
portal: https://github.com/GEMScienceTools/gem-global-active-faults
scale_tier: global
resolution: "fault-trace vectors compiled from source maps at scales from ~1:100,000 to 1:1,000,000"
data_types: [structural-vectors]
regions: [global]
coverage:
  name: Global (deforming continental regions; excludes Canada, Madagascar, Malay Archipelago)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson
    notes: >-
      GitHub-hosted, version-controlled. Harmonized (attribute-cleaned) geojson shown;
      repo also has /geojson/gem_active_faults.geojson (raw), plus /geopackage, /shapefile,
      /kml, /gmt directories at the same path depth. Clone or raw-download individual files.
formats: [geojson, geopackage, shp, kml]
license: CC BY-SA 4.0
added: 2026-08-11
verified: 2026-08-11
---

Styron & Pagani (2020)'s compilation — the standard open reference for active-fault traces
at continental-to-global scale, built for seismic hazard modeling (used in GEM's OpenQuake
stack). Community-maintained on GitHub; a live viewer sits at the OpenQuake hazard blog.
Coverage is not literally global — Canada, Madagascar and the Malay Archipelago are
explicit gaps. No slip-rate field is populated for most faults; check per-fault metadata
before assuming kinematic completeness.
