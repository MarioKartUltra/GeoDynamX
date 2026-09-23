---
id: nsw-seamless-geology
name: NSW Seamless Geology (MinView / SEED / DIGS)
operator: "[[Geological Survey of New South Wales]]"
portal: https://minview.geoscience.nsw.gov.au/
scale_tier: regional
resolution: "1:100,000 seamless compilation (source maps 1:25,000-1:250,000)"
data_types: [structural-vectors, geologic-map, borehole]
regions: [australia, new-south-wales]
coverage:
  name: New South Wales
  bbox: [140.9, -37.6, 153.7, -28.1]
fetch:
  - method: wfs
    url: "https://gs-seamless.geoscience.nsw.gov.au/geoserver/ows?service=WFS&acceptversions=2.0.0&request=GetCapabilities"
    notes: >-
      Per-lithotectonic-province and statewide layers, incl. geology:faults_*,
      geology-simplified:faults_and_shear_zones, and fold_axes_* — GetCapabilities
      confirmed live (155 KB response, real layer list).
  - method: http-download
    url: https://data.nsw.gov.au/data/dataset/nsw-seamless-geology
    notes: >-
      Packaged shp/gdb download plus WMS/WFS links and the MinView deep-link. The
      seed.nsw.gov.au mirror of this same listing sits behind an AWS WAF
      bot-challenge for scripted fetches (returns empty 202); this data.nsw.gov.au
      mirror does not.
  - method: manual
    url: https://digs.geoscience.nsw.gov.au/
    auth: free-registration
    notes: DIGS — borehole logs and the historical exploration/geological report archive
formats: [shp, fgdb]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

Three fronts on one compilation: MinView is the interactive viewer, SEED/data.nsw.gov.au
hosts the packaged statewide download, and the GeoServer WFS underneath exposes
per-province fault and fold-axis layers directly and by name. DIGS is the separate borehole
and historical-report archive — free but registration-walled. Note the friction: the
seed.nsw.gov.au dataset page itself is WAF-gated against scripted access; data.nsw.gov.au
mirrors the identical CKAN listing without the challenge.
