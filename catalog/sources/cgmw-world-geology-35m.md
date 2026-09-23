---
id: cgmw-world-geology-35m
name: CGMW Geological Map of the World at 1:35,000,000 (GIS)
operator: "[[Commission for the Geological Map of the World]]"
portal: https://www.ccgm.org/en/
scale_tier: global
resolution: "1:35,000,000"
data_types: [geologic-map, structural-vectors]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://www.ccgm.org/en/product/geological-map-of-the-world-at-1-35-m-sig/
    auth: license-agreement
    notes: >-
      2017 GIS restructure of the 2014 3rd-edition map (geodatabase + ArcGIS 10.0/10.5
      mxd, English and French). Not a self-serve download: the licence agreement must be
      downloaded, signed, and emailed to ccgm@sfr.fr before CGMW sends the file link.
formats: [fgdb, mxd]
license: Free, by signed CGMW/CCGM licence agreement (non-open; contact ccgm@sfr.fr for terms)
added: 2026-08-11
verified: 2026-08-11
---

CGMW/CCGM's flagship compilation — the closest thing to a single authoritative global
bedrock-geology-plus-major-structure map, with homogenized thrust fronts, subduction
zones, accretionary prisms and Cenozoic volcanics across the 2014/2020 edition. The
product page is live and browsable without login; the GIS itself sits behind a signed
licence agreement, not a click-through, so budget turnaround time before a fetch can
actually land a file.
