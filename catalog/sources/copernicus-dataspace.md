---
id: copernicus-dataspace
name: "Copernicus Data Space Ecosystem (STAC + OData)"
operator: "[[European Space Agency]]"
portal: https://dataspace.copernicus.eu/
scale_tier: global
resolution: "gateway — resolution per dataset"
data_types: [dem]
regions: [global]
coverage:
  name: Global (gateway)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: stac
    url: https://catalogue.dataspace.copernicus.eu/stac
    notes: >-
      Confirmed live: root catalog document returns real STAC JSON (id "cdse-stac").
      /stac/collections lists 418 collections (confirmed 2026-08-11), including
      cop-dem-glo-30-dged-cog / cop-dem-glo-90-dged-cog (`dem`) alongside the full
      Sentinel-1/2/3/5P/6 archives (SAR/optical/atmospheric — outside this catalog's
      data-type enum but reachable through the same API).
  - method: api
    url: "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$top=1"
    notes: "OData v1 product-metadata API — confirmed live, returns real JSON (Sentinel-3 product record in test query)."
  - method: manual
    url: https://dataspace.copernicus.eu/
    notes: "Copernicus Data Space Ecosystem portal — confirmed live (HTTP 200); browser UI at browser.dataspace.copernicus.eu sits over the same STAC catalog."
formats: [geotiff, jp2, safe, netcdf]
license: "Copernicus free/open-access license (attribution); Copernicus DEM carries the ESA/Airbus free license"
added: 2026-08-11
verified: 2026-08-11
---

**GATEWAY entry.** The Data Space Ecosystem's STAC and OData APIs are ESA's unified metadata
and product-search system fronting the entire Copernicus mission archive — this record is
about the system, distinct from the `copernicus-glo30` entry elsewhere in this catalog, which
records one specific DEM STAC recipe through the same API. Flagship holdings reachable
through it: Copernicus DEM GLO-30/GLO-90 global elevation (`dem`), plus the full Sentinel-1
SAR, Sentinel-2 optical, Sentinel-3 ocean/land, Sentinel-5P atmospheric, and Sentinel-6
altimetry archives — none of which map onto this catalog's geology-oriented data-type enum,
but all discoverable and fetchable through the same STAC/OData surface. Free registration is
required for direct data download; search itself is unauthenticated.
