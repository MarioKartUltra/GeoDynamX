---
id: nopims-offshore-wells-seismic
name: NOPIMS — National Offshore Petroleum Information Management System
operator: "[[National Offshore Petroleum Titles Administrator]]"
portal: https://www.ga.gov.au/nopims
scale_tier: national
resolution: "per-well and per-survey records; 2D/3D SEG-Y seismic volumes"
data_types: [seismic-reflection, borehole, core-sample]
regions: [australia]
coverage:
  name: Australian offshore basins (Commonwealth waters)
  bbox: [110.0, -47.0, 156.0, -8.0]
fetch:
  - method: manual
    url: https://public.neats.nopta.gov.au/nopims
    notes: "Primary data-discovery/delivery system for offshore well data, geophysical survey data, and physical samples (core, cuttings, fluids)"
  - method: manual
    url: https://www.ga.gov.au/nopims
    notes: "Program overview and contacts (ausgeodata@ga.gov.au general; ausgeosamples@ga.gov.au for physical samples); redeveloped jointly by NOPTA, WA DMIRS and Geoscience Australia"
license: "Described as openly accessible to industry, research organisations and the public for most holdings; no single blanket license statement found on the overview page"
added: 2026-08-11
verified: 2026-08-11
---

Australia's offshore-petroleum data system: well data, 2D/3D seismic surveys, and physical
samples (core, cuttings, fluids) for the Commonwealth offshore estate. Redeveloped from the legacy
NOPIMS into a NOPTA-hosted delivery portal in collaboration with Geoscience Australia and WA
DMIRS. The delivery portal is a JS-driven search/discovery app — most content isn't crawlable
statically, but the endpoint is live and the program overview confirms scope and contacts.
