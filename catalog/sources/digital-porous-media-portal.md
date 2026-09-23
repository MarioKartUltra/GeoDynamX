---
id: digital-porous-media-portal
name: Digital Porous Media Portal (formerly Digital Rocks Portal)
operator: "[[Texas Advanced Computing Center]]"
portal: https://digitalporousmedia.org/
scale_tier: sample
resolution: "micro-CT / imaging volumes of porous media, ~sub-um to mm voxels depending on dataset"
data_types: [microstructure, core-sample]
regions: [global]
coverage:
  name: Global (discrete sites)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://digitalporousmedia.org/
    notes: >
      rebrand of the Digital Rocks Portal (old DOI 10.17616/R3C64S superseded by
      10.17612/FGMN-D889); browse/search published projects, each dataset carries a DOI and
      downloadable raw + derived imagery plus specimen metadata (porosity, permeability, NMR,
      elastic properties, etc.)
license: per-dataset (contributor-assigned; CC licenses common)
added: 2026-08-11
verified: 2026-08-11
---

Repository of imaged porous-media/rock microstructure datasets (micro-CT, SEM, etc.) plus the
lab measurements tied to them — the standard target for digital rock physics work. Renamed
from "Digital Rocks Portal" in 2025; the old domain now 404s to a tombstone page pointing here.
Gotcha: login now requires MFA (since June 2025), and the portal is mid-migration ("beta testing
of the new portal") so some legacy dataset URLs may not have carried over cleanly. Cite by
per-dataset DOI, not the portal DOI.
