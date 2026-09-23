---
id: afead-active-faults
name: AFEAD — Active Faults of Eurasia Database
operator: "[[Geological Institute of the Russian Academy of Sciences]]"
portal: http://neotec.ginras.ru/index/english/database_eng.html
scale_tier: continental
resolution: "1:500,000 working (compilation) scale; 1:1,000,000 demonstration/sheet scale"
data_types: [structural-vectors]
regions: [eurasia]
coverage:
  name: Eurasia and adjacent seas
  bbox: [-25.0, 1.0, 180.0, 82.0]
fetch:
  - method: manual
    url: http://neotec.ginras.ru/index/database/sheets.html
    notes: >-
      No single combined download. Distributed as 4x6-degree sheets on the 1:1,000,000
      international map-sheet grid (nomenclature like F46, G39...), each sheet as raster
      JPG plus vector KMZ and SHP with full attribution (name, kinematics, slip rate rank,
      source citation). Site is HTTP-only — its HTTPS listener has a broken/self-signed
      cert, so use http:// explicitly.
formats: [shp, kmz, jpg]
license: CC BY 4.0
added: 2026-08-11
verified: 2026-08-11
---

Zelenin et al. (2022, ESSD) — 20,000+ Late Pleistocene/Holocene active-fault and
fault-zone objects across Eurasia, compiled from 612 published sources with justification
and estimated-parameter attributes per object. The AFEAD v.2022/2023 map viewer runs on
MapBox and YandexMap. Per-sheet distribution means a continental-scale pull means fetching
and mosaicking dozens of tiles — there is no bulk zip.
