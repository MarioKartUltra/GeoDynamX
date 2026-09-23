---
id: usgs-africa-geology-provinces
name: USGS Surficial Geology and Geologic Provinces of Africa (2000 World Energy Project)
operator: "[[U.S. Geological Survey]]"
portal: https://www.sciencebase.gov/catalog/item/60d0ff26d34e86b938aab404
scale_tier: continental
resolution: "reconnaissance-scale province/unit polygons; no fixed map scale (2000 World Energy Project compilation)"
data_types: [geologic-map]
regions: [africa]
coverage:
  name: Africa (continent, incl. Madagascar and Arabian Peninsula fringe)
  bbox: [-22.41, -38.57, 64.54, 40.14]
fetch:
  - method: http-download
    url: https://www.sciencebase.gov/catalog/file/get/60d0ff26d34e86b938aab404?f=__disk__7d%2F56%2Ff0%2F7d56f05682703babd743ee26112957e4ca83b5de
    notes: "Surficial geology (geo7_2ag) shapefile — direct .shp download confirmed live, no auth."
  - method: http-download
    url: https://www.sciencebase.gov/catalog/item/60d0ff40d34e86b938aab435
    notes: "Companion Geologic Provinces of Africa (prv7_2ag) item — same series, arcs+polygons+labels, shapefile set."
formats: [shp]
license: public domain (US federal)
added: 2026-08-11
verified: 2026-08-11
---

Two companion continent-scale USGS coverages from the same World Energy Project 2000 series
as the existing global geologic-provinces entry, but here at full continental extent rather
than assessed-provinces-only: surficial/quaternary geology (geo7_2ag) and structurally-defined
geologic provinces (prv7_2ag) for all of Africa. Reconnaissance generalization, not a
bedrock-unit map — offshore province limits mostly follow the 2000 m bathymetric contour.
Files sit as loose shapefile parts (.shp/.dbf/.prj/.sbn/.sbx/.shx/.xml) on ScienceBase, no
bundling zip; each file downloads individually from its own signed URL.
