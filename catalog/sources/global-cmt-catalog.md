---
id: global-cmt-catalog
name: Global CMT Catalog — Centroid Moment Tensors
operator: "[[Global CMT Project]]"
portal: https://www.globalcmt.org/
scale_tier: global
resolution: "event catalog, moment tensors for moderate-to-large earthquakes (~Mw>=5, smaller in well-recorded regions), 1976-present"
data_types: [earthquake-catalog]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec25.ndk.gz
    notes: >-
      Live-verified (HTTP 200, ~8.8 MB gzip): complete 1976-2025 catalog in
      NDK format. Monthly rolling updates at
      .../catalog/NEW_MONTHLY, quick (near-real-time) solutions at
      .../catalog/NEW_QUICK/.
  - method: manual
    url: https://www.globalcmt.org/CMTsearch.html
    notes: form-based search UI with 6 output formats (incl. CMTSOLUTION, GMT psmeca)
formats: [ndk]
license: free use; citation requested (see project citation guidelines)
added: 2026-08-11
verified: 2026-08-11
---

Successor to the Harvard CMT project, now at Lamont-Doherty. The reference global source
for focal mechanisms / moment tensors rather than hypocenter-only catalogs (ComCat, ISC) —
useful wherever fault-plane orientation is the quantity of interest. Bulk NDK files are
a plain static download, no auth.
