---
id: emodnet-geology-seismic
name: EMODnet Geology — Seismic and Multibeam Survey Data
operator: "[[EMODnet]]"
portal: https://emodnet.ec.europa.eu/en/geology
scale_tier: continental
resolution: "survey-index/line-location metadata, federated from national geological surveys; underlying trace data resolution varies per contributor"
data_types: [seismic-reflection, bathymetry]
regions: [europe]
coverage:
  name: European seas
  bbox: [-44.0, 24.0, 45.0, 82.0]
fetch:
  - method: wms
    url: https://drive.emodnet-geology.eu/geoserver/wms?service=WMS&request=GetCapabilities
    notes: Verified live 2026-08-11 (200 OK GetCapabilities). Layers are organised per contributing national survey workspace (e.g. bgr, bgs, tno); see the geoviewer to browse which layer covers which area.
  - method: manual
    url: https://emodnet.ec.europa.eu/geoviewer/
    notes: Central map viewer, "Seismic and Multibeam Survey Data" theme among others (seabed substrates, geology, hazards, minerals).
  - method: manual
    url: https://emodnet.ec.europa.eu/geonetwork/srv/eng/catalog.search
    notes: Metadata catalogue for discovering individual contributing datasets/services.
formats: [wms, shp, geotiff]
license: Varies by contributing national geological survey — check per-dataset metadata in the catalogue
added: 2026-08-11
verified: 2026-08-11
---

EMODnet Geology's cross-national index of where seismic-reflection and multibeam surveys
exist, harmonised from national geological surveys' holdings alongside its seabed-substrate,
Quaternary-geology and geohazard themes. Gotcha shared with EGDI: this is federation
metadata/line-location, not a single raw-SEG-Y download — the actual trace data for a given
survey typically still lives at the contributing national survey and is reached by following
the catalogue entry out, not by pulling from EMODnet's own service.
