---
id: scar-geomap-antarctica-geology
name: GeoMAP — Continent-Wide Geological Dataset of Antarctica
operator: "[[GNS Science]]"
portal: https://www.gns.cri.nz/data-and-resources/geomap-geological-mapping-of-antarctica/
scale_tier: continental
resolution: "1:250,000-class unified compilation (locally finer where source maps allow)"
data_types: [geologic-map, structural-vectors]
regions: [antarctica]
coverage:
  name: Antarctica (exposed bedrock and surficial geology)
  bbox: [-180.0, -90.0, 180.0, -60.0]
fetch:
  - method: arcgis-rest
    url: https://gis.gns.cri.nz/server/rest/services/SCAR_GeoMAP/ATA_SCAR_GeoMAP_Geology/MapServer?f=json
    notes: >-
      Confirmed live ArcGIS Server 11.5, EPSG:3031. 19 layers incl. mapped faults (polyline),
      chrono/lithostratigraphic units at >500k and <500k scale, simple geology/lithology
      polygons, source-map index, and a data-quality-assessment layer.
  - method: http-download
    url: https://doi.pangaea.de/10.1594/PANGAEA.951482
    notes: "v.2022-08 release — 99,080 polygons as fgdb (115 MB), gpkg (187 MB), or KMZ (298 MB)."
formats: [fgdb, gpkg, kmz]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

SCAR GeoMAP Action Group + GNS Science's unification of legacy Antarctic geological
quadrangle maps into one harmonized GIS — the actual bedrock-geology-and-faults layer that
ADD and Bedmap lack. Classification mixes chronostratigraphic and lithostratigraphic schemes
by necessity (compiled from decades of disparate source maps), so unit labels aren't fully
uniform across the continent. The PANGAEA download is a single multi-hundred-MB archive;
the ArcGIS service is the lighter-weight route for querying a specific area.
