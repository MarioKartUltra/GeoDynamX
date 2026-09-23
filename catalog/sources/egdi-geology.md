---
id: egdi-geology
name: EGDI — European Geological Data Infrastructure (geology/structure layers)
operator: "[[EuroGeoSurveys]]"
portal: https://www.europe-geology.eu/
scale_tier: continental
resolution: "variable — federated national-survey layers, typically 1:50,000 to 1:1,000,000"
data_types: [geologic-map, structural-vectors]
regions: [europe]
coverage:
  name: Europe
  bbox: [-25.0, 34.0, 45.0, 72.0]
fetch:
  - method: manual
    url: https://www.europe-geology.eu/data-tools/map-services-and-layers/
    notes: >-
      Federation portal, not a single dataset: this page lists every layer in the EGDI
      Map Viewer with metadata/service-status flags, each backed by its own national
      survey's WMS/WFS. Discover a layer here, then hit its GetCapabilities directly.
  - method: wfs
    url: https://ogc2.bgs.ac.uk/cgi-bin/BGS_OGE_Bedrock_and_Surface_Geology/ows?service=WFS&request=GetCapabilities
    notes: One concrete example member service — BGS bedrock/surface geology WFS, indexed via EGDI's OneGeology-Europe layer.
formats: [shp, wms, wfs]
license: Varies by contributing national geological survey — check per-dataset metadata in the EGDI catalogue
added: 2026-08-11
verified: 2026-08-11
---

The successor to OneGeology-Europe: EuroGeoSurveys' harmonised access point over national
geological surveys' bedrock geology, structural, mineral-resource and other thematic
layers, browsable at the EGDI Map Viewer (maps.europe-geology.eu). Same friction as any
federation portal — there is no single "download all of Europe" file; each layer is a
separate WMS/WFS from a separate host, with independently variable scale, currency and
licence. The map-services-and-layers page is the map of the map, i.e. where to find which
service is currently live.
