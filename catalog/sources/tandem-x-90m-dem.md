---
id: tandem-x-90m-dem
name: TanDEM-X 90m Digital Elevation Model (TDM90)
operator: "[[German Aerospace Center]]"
portal: https://geoservice.dlr.de/web/dataguide/tdm90/
scale_tier: global
resolution: "90 m posting (3 arc-second in latitude)"
data_types: [dem]
regions: [global]
coverage:
  name: Global land, pole to pole
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://download.geoservice.dlr.de/TDM90/files/
    auth: free-registration
    notes: >-
      Confirmed login-gated: an unauthenticated request to this path redirects to the DLR
      EOC "TDM Login" page (verified 2026-08-11). Self-register at
      https://sso.eoc.dlr.de/tdm90/selfservice/, then browse or script (wget/cURL/aria2)
      the file tree of per-tile ZIPs. A map-based selector is also at
      https://download.geoservice.dlr.de/TDM90/. (Note: a generic-looking
      geoservice.dlr.de WCS endpoint at /eoc/land/wcs/tdm90 exists but serves unrelated
      German land-cover layers, not this DEM — checked and ruled out.)
formats: [geotiff]
license: DLR scientific-use license (free, non-exclusive, non-commercial; commercial licensing via Airbus Defence and Space)
added: 2026-08-11
verified: 2026-08-11
---

Unedited single-pass InSAR digital surface model (not bare-earth — includes canopy and
buildings) from the TanDEM-X radar mission, covering all ~150M km² of Earth's
landmasses at 3 arc-second spacing with ~1 m absolute vertical accuracy. Ships with
seven auxiliary layers (height error, water mask, consistency mask, layover/shadow mask,
etc.) per tile — check those before trusting DEM values in radar-shadow or water cells.
Free for science but requires DLR account registration; commercial use needs an Airbus
license.
