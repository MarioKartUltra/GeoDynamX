---
id: emodnet-bathymetry
name: EMODnet Bathymetry — Digital Terrain Model
operator: "[[EMODnet]]"
portal: https://emodnet.ec.europa.eu/en/bathymetry
scale_tier: continental
resolution: "1/16 x 1/16 arc-minute DTM (~115 m at European latitudes); 200+ coastal high-resolution DTMs down to 1/512 arc-minute"
data_types: [bathymetry]
regions: [europe]
coverage:
  name: European seas (North Sea, NE Atlantic, Baltic, Mediterranean, Black Sea, Arctic/Barents)
  bbox: [-44.0, 24.0, 45.0, 82.0]
fetch:
  - method: wcs
    url: https://ows.emodnet-bathymetry.eu/wcs?service=WCS&request=GetCapabilities
    notes: Verified live 2026-08-11 (200 OK GetCapabilities).
  - method: wms
    url: https://ows.emodnet-bathymetry.eu/wms?service=WMS&request=GetCapabilities
    notes: Verified live 2026-08-11.
  - method: wms
    url: https://tiles.emodnet-bathymetry.eu/
    notes: EMODnet Bathymetry World Base Layer tile service (WMTS), verified live 2026-08-11.
  - method: manual
    url: https://emodnet.ec.europa.eu/geoviewer/
    notes: Central map viewer with area-select download; DOI-cited release, 2024 DTM current as of this check.
formats: [geotiff, netcdf, esri-ascii, xyz, csv]
license: Free reuse with attribution (EMODnet Bathymetry Consortium DOI citation)
added: 2026-08-11
verified: 2026-08-11
---

The European counterpart to GEBCO/GMRT: a harmonised DTM built from national-survey
contributions, in both Lowest-Astronomical-Tide and Mean-Sea-Level vertical references, with
a source-reference layer and quality-index map per cell. Coastal areas get much
higher-resolution nested DTMs (down to 1/512 arc-min) where contributing surveys support it —
the headline 1/16 arc-min figure is the coarse open-ocean baseline, not what's available
everywhere. OGC services (WMS/WFS/WCS/WMTS) are live and unauthenticated.
