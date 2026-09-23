---
id: gns-nz-active-faults
name: New Zealand Active Faults Database (NZAFD)
operator: "[[GNS Science]]"
portal: https://www.gns.cri.nz/data-and-resources/new-zealand-active-faults-database/
scale_tier: national
resolution: "1:250,000 (generalized AF250 traces); source mapping varies by fault"
data_types: [structural-vectors]
regions: [oceania, new-zealand]
coverage:
  name: New Zealand
  bbox: [166.4, -47.3, 178.6, -34.4]
fetch:
  - method: wfs
    url: https://maps.gns.cri.nz/gns/ows?request=GetCapabilities&service=wfs&version=1.0.0
    notes: "confirmed live (200, full capabilities doc); layer gns:af250_faults_pg = generalized active-fault traces"
  - method: wms
    url: https://gis.gns.cri.nz/server/rest/services/Active_Faults/NZActiveFaultDatasets/MapServer
    notes: "ArcGIS MapServer backing the interactive webmap at data.gns.cri.nz/af/"
formats: [shp, gml, kml]
license: CC BY 3.0 NZ
added: 2026-08-11
verified: 2026-08-11
---

Nationwide onshore active-fault traces (surface rupture in the last 125,000 years) maintained by GNS
Science, the model single-purpose structural-vector database. Served both as a WFS (queried live,
returns real fault-trace features) and an ArcGIS MapServer behind the public webmap. Free, CC BY 3.0 NZ,
no registration wall found. Coverage is onshore-only — offshore/submarine faults are a separate GNS
product not confirmed here.
