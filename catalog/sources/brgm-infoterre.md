---
id: brgm-infoterre
name: BRGM InfoTerre — BD Charm-50 harmonized geological maps
operator: "[[Bureau de Recherches Géologiques et Minières]]"
portal: https://infoterre.brgm.fr/
scale_tier: national
resolution: "1:50,000 (BD Charm-50); 1:250,000; 1:1,000,000"
data_types: [structural-vectors, geologic-map, borehole]
regions: [europe, france]
coverage:
  name: Metropolitan France
  bbox: [-5.2, 41.3, 9.6, 51.1]
fetch:
  - method: wms
    url: https://geoservices.brgm.fr/geologie?service=WMS&request=GetCapabilities
    notes: "GetCapabilities confirmed live (380KB response); layers include SCAN_D_GEOL50, SCAN_F_GEOL250, SCAN_F_GEOL1M, GEOLOGIE_OUTRE_MER"
  - method: http-download
    url: https://infoterre.brgm.fr/page/telechargement-cartes-geologiques
    notes: "vectorized/harmonized 1:50,000 departmental sheets (BD Charm-50) as shapefile, incl. faults and contacts"
  - method: wfs
    url: https://infoterre.brgm.fr/page/geoservices-ogc
    notes: "OGC WFS 1.0 download service, INSPIRE geology theme; also boreholes (BSS) and BSSEAU groundwater layers"
license: "Licence Ouverte / Etalab 2.0 — free for commercial and non-commercial reuse, attribution to BRGM + last-update date required"
added: 2026-08-11
verified: 2026-08-11
---

BRGM's InfoTerre is France's national geoscience portal: BD Charm-50, the vectorized and harmonized
departmental 1:50,000 geological map covering the whole of metropolitan France (plus overseas sheets),
with structural contacts and fault layers, alongside the Banque du Sous-Sol (BSS) borehole database.
WMS/WFS are OGC-standard and INSPIRE-conformant; everything is released under the fully open Licence
Ouverte / Etalab 2.0 as part of BRGM's "science ouverte" push — no registration wall, but note that
the data must not be altered in a way that distorts its meaning (Etalab condition).
