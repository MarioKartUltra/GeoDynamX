---
id: ingemmet-geocatmin-peru
name: GEOCATMIN — INGEMMET Geological & Mining Cadastral Information System (Peru)
operator: "[[INGEMMET]]"
portal: https://geocatmin.ingemmet.gob.pe/geocatmin/
scale_tier: national
resolution: "1:50,000 to 1:1,000,000 (geologic map series); airborne magnetics at survey resolution"
data_types: [geologic-map, structural-vectors, magnetics]
regions: [south-america, peru]
coverage:
  name: Peru
  bbox: [-81.4, -18.4, -68.7, 0.0]
fetch:
  - method: arcgis-rest
    url: https://geocatmin.ingemmet.gob.pe/arcgis/rest/services?f=json
    notes: "confirmed live ArcGIS Server 10.91; folders incl. BDGEOCIENTIFICA, DGAR, GEOPROCESO; services incl. SERV_AEROMAGNETIICO (ImageServer), SERV_ANOMALIA_ESPECTRAL, SERV_ATLAS_GEOQUIMICO"
formats: [shp, tiff]
license: "INGEMMET Licencia de Uso — reuse/derivative works permitted with mandatory 'CITE INGEMMET' attribution; no software redistribution"
added: 2026-08-11
verified: 2026-08-11
---

Peru's geological/mining survey exposes its bedrock geology, structural, geochemical, and aeromagnetic
holdings through a public ArcGIS REST service tree behind the GEOCATMIN viewer — dozens of named
services confirmed live, including an ImageServer for aeromagnetic data. Spanish-language only; the
licence permits reuse and derivatives with mandatory attribution but forbids redistributing the
software/interface itself. Gravity data was not confirmed present in the folders inspected.
