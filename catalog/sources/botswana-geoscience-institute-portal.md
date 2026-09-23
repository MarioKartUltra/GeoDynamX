---
id: botswana-geoscience-institute-portal
name: Botswana Geoscience Institute — Geological Portal
operator: "[[Botswana Geoscience Institute]]"
portal: https://www.bgi.org.bw/
scale_tier: national
resolution: "variable; national geological mapping, aeromagnetic survey grids, drill-core repository records"
data_types: [geologic-map, magnetics, borehole]
regions: [africa, botswana]
coverage:
  name: Botswana
  bbox: [19.9, -26.9, 29.4, -17.8]
fetch:
  - method: manual
    url: https://www.bgi.org.bw/botswana-geological-portal
    notes: >-
      Page confirmed live (HTTP 200 with a modern browser UA; default/no-UA requests get a
      403 from the front-end WAF). Describes an interactive GIS web-map ("Botswana Geoscience
      Portal", link given as geos.bgi.org.bw/GDP/Search) plus BGI Drill Core Repositories and
      a paid Data/Information Price List. The GDP GIS subdomain itself timed out on repeated
      checks during this verification — treat it as gated/intermittent, not confirmed live.
formats: [shp, gdb]
license: fee-based (BGI Data/Information Price List); some releases (e.g. aeromagnetic survey announcements) noted as free
added: 2026-08-11
verified: 2026-08-11
---

Botswana's national geoscience data agency (successor to the Dept. of Geological Survey),
recently announced a free release of Nossop-Ncojane aeromagnetic survey data alongside its
usual mapping/core-repository/borehole holdings. Most data is fee-gated per the published
price list rather than open download, and the actual web-GIS endpoint (geos.bgi.org.bw) was
unreachable during this verification pass even though the marketing/description page that
links to it is live — expect to contact BGI directly to actually pull a layer.
