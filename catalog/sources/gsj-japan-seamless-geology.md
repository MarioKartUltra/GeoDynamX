---
id: gsj-japan-seamless-geology
name: Seamless Digital Geological Map of Japan V2 (1:200,000)
operator: "[[Geological Survey of Japan]]"
portal: https://gbank.gsj.jp/seamless/
scale_tier: national
resolution: "1:200,000"
data_types: [geologic-map, structural-vectors]
regions: [asia, japan]
coverage:
  name: Japan
  bbox: [122.9, 24.0, 153.99, 45.6]
fetch:
  - method: api
    url: https://gbank.gsj.jp/seamless/v2/api/1.2.1/
    notes: "confirmed live web API — PNG map tiles (z/y/x), legend as JSON/CSV/HTML, map export as PNG/KMZ"
  - method: http-download
    url: https://gbank.gsj.jp/seamless/
    notes: "nationwide vector download (shapefile, KML) plus per-sheet 1:200,000 sections"
formats: [shp, kml, kmz, png]
license: "Government Standard Terms of Use v2.0 (Japan) — free reuse with attribution"
added: 2026-08-11
verified: 2026-08-11
---

AIST/GSJ's unified 1:200,000 bedrock geology for the whole of Japan, stitched from the historical paper
sheet series into one seamless product. Vector downloads (shapefile/KML) cover lithology and geologic
boundaries/structural lines; a documented tile+legend+export API backs the web viewer. Interface is
Japanese-only in places (page text, some legend fields); terms of use require attribution but no
registration observed for the API or bulk vector download.
