---
id: bgr-geologische-uebersichtskarten
name: BGR Geological Overview Maps of Germany (GÜK200/GÜK250)
operator: "[[Bundesanstalt für Geowissenschaften und Rohstoffe]]"
portal: https://geoportal.bgr.de/
scale_tier: national
resolution: "1:200,000 (GÜK200) / 1:250,000 (GÜK250)"
data_types: [geologic-map]
regions: [europe, germany]
coverage:
  name: Germany
  bbox: [5.8, 47.2, 15.1, 55.1]
fetch:
  - method: wms
    url: https://services.bgr.de/wms/geologie/guek200/?service=WMS&request=GetCapabilities
    notes: "GetCapabilities confirmed live (200); compiled with the state geological services (SGD)"
  - method: manual
    url: https://www.bgr.bund.de/DE/Themen/Sammlungen-Grundlagen/GG_geol_Info/Karten/Deutschland/deutschland_node.html
    notes: "GÜK250 successor product; shapefile/GML download via BGR Produktcenter (produktcenter.bgr.de), INSPIRE-conformant version also served as WMS"
license: "Data licence Germany – Attribution – Version 2.0 (dl-de/by-2-0), per GeoNutzV — free for commercial and non-commercial use, no licence agreement needed, source citation required"
added: 2026-08-11
verified: 2026-08-11
---

BGR's federal-scale bedrock geology, compiled jointly with the 16 state geological services (SGD) into
seamless national coverage at 1:200,000 (GÜK200, being superseded) and 1:250,000 (GÜK250). Served as WMS
from geoportal.bgr.de and downloadable as shapefile/GML via the BGR Produktcenter. Free under Germany's
federal open-geodata regulation (GeoNutzV) — no per-dataset licence needed, just attribution. The WMS layer
names returned by GetCapabilities are undescriptive numeric tile IDs rather than named feature classes, so
treat this as a compiled raster-map product rather than a distinct fault/structural vector layer; BGR's own
gravity/magnetics compilations are separate products not covered by this endpoint.
