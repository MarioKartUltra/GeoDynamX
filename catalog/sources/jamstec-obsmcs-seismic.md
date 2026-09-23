---
id: jamstec-obsmcs-seismic
name: JAMSTEC Multi-Channel Seismic and OBS Database
operator: "[[JAMSTEC]]"
portal: https://www.jamstec.go.jp/obsmcs_db/e/
scale_tier: regional
resolution: "survey-level multi-channel seismic (MCS) reflection lines and ocean-bottom-seismometer (OBS) deployments"
data_types: [seismic-reflection]
regions: [asia, japan, northwest-pacific]
coverage:
  name: Japan margin and NW Pacific (Japan/Kuril Trench, Nankai-Boso, Izu-Bonin-Mariana, Ryukyu, Japan Sea, Western Pacific)
  bbox: [120.0, 10.0, 165.0, 55.0]
fetch:
  - method: manual
    url: https://www.jamstec.go.jp/obsmcs_db/e/
    notes: >-
      Verified live 2026-08-11. Browse by method (MCS/OBS), year, or named area
      (survey/list_mcs.html, list_obs.html, list_area.html?area=...).
  - method: manual
    url: https://www.jamstec.go.jp/obsmcs_db/form/obsmcs_db_entry_e/index.html
    notes: Data request form — MCS/OBS data is not directly downloadable; access is granted per request for academic use.
native_metadata:
  - format: DOI landing page
    url: https://www.jamstec.go.jp/datadoi/doi/10.17596/0002069.html
license: "Free for scientific/educational use per JAMSTEC's Basic Policy on the Handling of Data and Samples (jamstec.go.jp/e/database/data_policy.html); industrial use is chargeable and requires separate arrangement"
added: 2026-08-11
verified: 2026-08-11
---

JAMSTEC's catalog of its own research-vessel multi-channel seismic reflection and
ocean-bottom-seismometer surveys around Japan and the NW Pacific (Nankai Trough, Japan/Kuril
Trench, Izu-Bonin-Mariana, Ryukyu). Friction: this is a request-gated archive, not an
open-download one — browsing and searching by area/method/year is free, but obtaining the
actual MCS/OBS data requires submitting the linked request form and is restricted to
academic use (a DOI is issued for citation once granted).
