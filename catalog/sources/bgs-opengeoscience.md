---
id: bgs-opengeoscience
name: BGS OpenGeoscience — Digital Geological Map of Great Britain (DiGMapGB)
operator: "[[British Geological Survey]]"
portal: https://www.bgs.ac.uk/geological-data/opengeoscience/
scale_tier: national
resolution: "1:625,000 to 1:10,000 (DiGMapGB series)"
data_types: [structural-vectors, geologic-map, borehole]
regions: [europe, united-kingdom]
coverage:
  name: Great Britain (onshore)
  bbox: [-8.65, 49.8, 1.8, 60.9]
fetch:
  - method: wms
    url: https://map.bgs.ac.uk/arcgis/services/BGS_Detailed_Geology/MapServer/WMSServer?service=WMS&request=GetCapabilities
    notes: "GetCapabilities confirmed live; layers include BGS.50k.Bedrock, BGS.50k.Superficial.deposits, BGS.50k.Linear.features (faults/structural lines), BGS.50k.Mass.movement, BGS.50k.Artificial.ground"
  - method: http-download
    url: https://www.bgs.ac.uk/datasets/bgs-geology-625k/
    notes: "statewide 1:625k bedrock+superficial+linear-features as shp/GeoPackage, free under OGL"
  - method: wfs
    url: https://www.bgs.ac.uk/technologies/web-services/web-feature-services-wfs/
    notes: "GeoSciML v4.1 / INSPIRE-conformant download service; portal page listing WFS endpoints per product"
license: "OGL v3 (BGS Geology 625k, free for commercial/research/public use with attribution); BGS Geology 50k (DiGMapGB) is free to view but carries a commercial licence fee (~£0.23/km²) for licensed use"
added: 2026-08-11
verified: 2026-08-11
---

The Digital Geological Map of Great Britain, BGS's flagship packaged-and-serviced product line: bedrock,
superficial deposits, artificial ground, mass movement, and a dedicated linear-features layer carrying faults
and other structural lines, at scales from 1:625k down to 1:10k. WMS/WFS are live and INSPIRE-conformant;
the 1:625k tier is fully open (OGL), while the detailed 1:50k tier (DiGMapGB-50) is free for non-commercial
use but has a per-km² licence fee for commercial reuse — the one friction point to plan around. Onshore
borehole records are indexed through the same OpenGeoscience/GeoIndex family.
