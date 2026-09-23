---
id: merit-dem
name: MERIT DEM
operator: "[[University of Tokyo]]"
portal: https://global-hydrodynamics.github.io/MERIT_DEM/
scale_tier: global
resolution: "90 m posting (3 arc-second)"
data_types: [dem]
regions: [global]
coverage:
  name: Global land, 90°N-60°S
  bbox: [-180.0, -60.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://global-hydrodynamics.github.io/MERIT_DEM/
    auth: free-registration
    notes: >-
      Registration via Google Form (https://goo.gl/forms/f3AlXftlXyODDxj32) yields an
      emailed password gating a Dropbox-hosted download tree of 5°x5° GeoTIFF tiles
      bundled into 30°x30° packages. Current release v1.0.3 (2018). Lab page moved here
      from its old hydro.iis.u-tokyo.ac.jp address (redirect confirmed live 2026-08-11).
formats: [geotiff]
license: "CC BY-NC 4.0 (non-commercial) or ODbL 1.0 (commercial, share-alike) - user's choice"
added: 2026-08-11
verified: 2026-08-11
---

Removes speckle noise, tree-height bias, and stripe/absolute-bias errors from
SRTM3/AW3D/ViewfinderPanoramas by multi-error-removal compositing (Yamazaki et al. 2017)
— a bare-earth-oriented alternative to raw SRTM at the same 3 arc-second grid. Access is
gated behind a Google Form plus an emailed password rather than a self-service account,
so onboarding has a manual, non-instant step.
