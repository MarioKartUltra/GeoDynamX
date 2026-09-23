---
id: nevada-nbmg-geology-faults
name: Nevada Bureau of Mines and Geology — Statewide Geology & Faults
operator: "[[Nevada Bureau of Mines and Geology]]"
portal: https://nbmg.unr.edu/
scale_tier: regional
resolution: "1:500,000 (geology); compiled fault traces at source-map scale"
data_types: [structural-vectors, geologic-map]
regions: [north-america, united-states, nevada]
coverage:
  name: Nevada
  bbox: [-120.0, 35.0, -114.0, 42.0]
fetch:
  - method: arcgis-rest
    url: https://gisweb.unr.edu/nbmg/rest/services/Geology/NV_500k_Geology/MapServer
    notes: Stewart & Carlson 1:500,000 Geologic Map of Nevada (OFR 03-66), units plus structure/contact data
  - method: arcgis-rest
    url: https://gisweb.unr.edu/nbmg/rest/services/Geology/Faults/MapServer
    notes: '"Historical Ruptures" and "Quaternary Faults by Age" polylines, adapted from NBMG Map 167 and USGS Qfaults'
formats: [shp]
license: NBMG (University of Nevada, Reno) — free public streaming/download; no separate open-data license statement found
added: 2026-08-11
verified: 2026-08-11
---

NBMG's ArcGIS Server (gisweb.unr.edu/nbmg/rest/services) hosts a full Geology folder — statewide
1:500k geology, dedicated fault layers, geochemistry, geochronology, and several project-specific
maps (Rhyolite Ridge, Clayton Valley, Jersey Summit). The fault service explicitly derives from and
cross-references both NBMG's own Map 167 and the USGS Quaternary Fault and Fold Database, so treat
it as a locally-refined subset rather than an independent compilation.
