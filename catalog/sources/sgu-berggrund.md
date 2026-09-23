---
id: sgu-berggrund
name: SGU Berggrund — Swedish bedrock geology and deformation zones
operator: "[[Geological Survey of Sweden]]"
portal: https://www.sgu.se/en/products/geological-data/berggrund--geologisk-data/bedrock/
scale_tier: national
resolution: "1:50,000-1:250,000; 1:1,000,000 (bedrock overview)"
data_types: [structural-vectors, geologic-map]
regions: [europe, sweden]
coverage:
  name: Sweden
  bbox: [10.9, 55.3, 24.2, 69.1]
fetch:
  - method: wms
    url: https://resource.sgu.se/service/wms/130/berggrund_1M?service=WMS&request=GetCapabilities
    notes: "GetCapabilities confirmed live; layers SE.GOV.SGU.BERGGRUND_NA10 (bedrock), BERGGRUND_NA10_DFZ (deformation zones/faults), BERGGRUND_NA10_LINJER (structural lines)"
  - method: wms
    url: https://resource.sgu.se/service/wms/130/berggrund-50-250-tusen?service=WMS&request=GetCapabilities
    notes: "detailed 1:50,000-1:250,000 bedrock WMS"
  - method: http-download
    url: https://www.sgu.se/en/products/geological-data/berggrund--geologisk-data/bedrock/
    notes: "GeoPackage/API downloads; OGC API - Features supersedes the older WFS interface"
license: "CC0 — SGU made all geological data open and free of charge from June 2024 (Sweden's first agency to do so under the EU open-data directive)"
added: 2026-08-11
verified: 2026-08-11
---

Sweden's bedrock geology at three nested scales, with an explicit deformation-zone (DFZ) vector layer
distinct from the lithology polygons — a clean structural-vectors match confirmed directly in the WMS
GetCapabilities layer list. SGU became fully open in mid-2024 (CC0, no fees, no registration), a recent
and significant friction removal worth noting for anyone who remembers the old paid-product era.
