---
id: copernicus-glo30
name: Copernicus DEM GLO-30
operator: "[[European Space Agency]]"
portal: https://dataspace.copernicus.eu/
scale_tier: global
resolution: 30 m posting (GLO-30)
data_types: [dem]
regions: [global]
coverage:
  name: Global (land)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: http-download
    url: https://copernicus-dem-30m.s3.amazonaws.com/
    notes: AWS Open Data mirror, tiled GeoTIFF, no auth
  - method: stac
    url: https://catalogue.dataspace.copernicus.eu/stac
    auth: free-registration
formats: [geotiff]
license: ESA/Airbus free license (attribution)
added: 2026-08-11
verified: 2026-08-11
---

The default global elevation baseline. TanDEM-X-derived; edited water surfaces. The AWS
mirror is the friction-free path; the Copernicus Data Space STAC API is the canonical one.
