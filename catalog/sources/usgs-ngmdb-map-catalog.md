---
id: usgs-ngmdb-map-catalog
name: USGS National Geologic Map Database — Map Catalog & Mapview
operator: "[[U.S. Geological Survey]]"
portal: https://ngmdb.usgs.gov/ngmdb/ngmdb_home.html
scale_tier: national
resolution: "varies by contributing map, 1:12,000-1:1,000,000"
data_types: [geologic-map]
regions: [north-america, united-states]
coverage:
  name: United States (incl. Alaska, Hawaii, territories)
  bbox: [-179.15, 18.91, -66.87, 71.44]
fetch:
  - method: manual
    url: https://ngmdb.usgs.gov/mapview/
    notes: interactive search/browse of 90,000+ published geologic maps; many entries link to GeMS GIS downloads, others to scanned sheets only
formats: [pdf, gdb, shp, tiff]
license: Public domain (USGS); maps contributed by ~630 state/university/private agencies may carry separate terms
added: 2026-08-11
verified: 2026-08-11
---

The federal clearinghouse mandated by the 1992 Geologic Mapping Act — a catalog/index rather than a
single seamless dataset. Coverage is uneven: some map records have GeMS-format GIS packages attached,
many older ones are scan-only. Best used as a discovery layer pointing back to state surveys (several
of which are catalogued separately in this cell) rather than as a standalone vector source. Site was
mid-migration to new infrastructure at verification time; a banner warns of possible disruptions.
