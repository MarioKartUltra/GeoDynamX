---
id: usgs-namss
name: USGS National Archive of Marine Seismic Surveys (NAMSS)
operator: "[[USGS]]"
portal: https://walrus.wr.usgs.gov/namss/
scale_tier: national
resolution: "survey-level 2D/3D multichannel and single-channel seismic reflection, mostly SEG-Y; some pre-digital surveys are scanned profile images only"
data_types: [seismic-reflection]
regions: [north-america, united-states]
coverage:
  name: US Outer Continental Shelf margins (Pacific, Gulf of Mexico, Atlantic, Alaska)
  bbox: [-180.0, 15.0, -60.0, 72.0]
fetch:
  - method: manual
    url: https://walrus.wr.usgs.gov/namss/
    notes: >-
      Interactive map/filter search (survey ID, year, data type, geographic click);
      verified live 2026-08-11. Data type filter distinguishes 2D/3D multichannel vs
      single-channel seismic.
  - method: wms
    url: https://walrus.wr.usgs.gov/namss/wms?request=GetCapabilities&service=WMS&version=1.1.1
    notes: Verified live 2026-08-11 (survey-footprint WMS layer).
formats: [segy, tiff]
license: Public domain (US federal) — non-exclusive-use G&G permit data, no restrictions on usage or publication once archived
added: 2026-08-11
verified: 2026-08-11
---

Originally seeded from Outer Continental Shelf geological-and-geophysical (G&G) permit data
under 30 CFR Parts 551/580 — once a permit's exclusive-use period expires, the survey becomes
public domain and lands here. Current stewardship is the USGS Pacific Coastal and Marine
Science Center's West Coast/Alaska project, but holdings include Gulf of Mexico and Atlantic
survey codes too. Most data is SEG-Y; older surveys may have only a scanned image of the
reflection profile with no digital trace file, so "SEG-Y available" cannot be assumed per
survey — check each record's format flag.
