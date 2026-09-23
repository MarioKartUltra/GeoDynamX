---
id: v3geo
name: V3Geo — Virtual 3D Outcrop Model Repository
operator: "[[University of Aberdeen]]"
portal: https://v3geo.com/
scale_tier: local
resolution: "cm-scale textured meshes (photogrammetry/lidar-derived virtual outcrops)"
data_types: [outcrop-model]
regions: [global]
coverage:
  name: Global (discrete sites)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://v3geo.com/
    notes: >
      cloud-based web viewer streams tiled 3D meshes (photogrammetry/laser-scan derived);
      browse/search by site, view interactively with measurement tools; per-model download
      terms set by contributor, no bulk API
license: varies per model (contributor-assigned Creative Commons; check each entry)
added: 2026-08-11
verified: 2026-08-11
---

Cloud repository purpose-built for virtual 3D outcrop models — photogrammetry and lidar meshes
streamed to a web viewer with embedded measurement tools, no specialist software required.
Coverage spans microscopic/hand-sample to multi-kilometre terrain, but outcrop-scale virtual
field sites dominate the catalog. Public models carry contributor-assigned CC licenses; the
homepage is a JS single-page app so machine-readability of the catalog itself is poor — treat
as browse-only, no bulk metadata API. Individual models often carry a DOI from the underlying
publication (Buckley et al., 2022, *Geoscience Communication*).
