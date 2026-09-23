---
id: aer-alberta-well-core-data
name: AER Alberta — well data and Core Research Centre
operator: "[[Alberta Energy Regulator]]"
portal: https://www.aer.ca/data-and-performance-reports/activity-and-data
scale_tier: regional
resolution: "per-well drilling data; per-well core/sample inventory"
data_types: [borehole, core-sample]
regions: [north-america, canada, alberta]
coverage:
  name: Alberta
  bbox: [-120.5, 48.7, -110.0, 60.0]
fetch:
  - method: manual
    url: https://www.aer.ca/data-and-performance-reports/activity-and-data/lists-and-activities/general-well-data
    notes: "Daily-updated General Well Data report — basic drilling data for every oil, gas, oil sands and water well in Alberta (no production data); xlsx"
  - method: manual
    url: https://www1.aer.ca/ProductCatalogue/index.html
    notes: "Products and Services Catalogue — bulk All-Alberta well-data file; returned HTTP 403 to a bare curl request, loads in-browser"
  - method: manual
    url: https://www.aer.ca/about-aer/research-facilities/core-research-centre
    notes: "Core Research Centre — physical core/cuttings repository; viewing by appointment, ordering procedure documented separately"
formats: [xlsx, csv]
license: "Open Government Licence – Alberta for published open datasets; Core Research Centre physical-sample access follows separate AER procedures/fees"
added: 2026-08-11
verified: 2026-08-11
---

Alberta's petroleum regulator publishes a daily General Well Data report (basic drilling data,
every well type, no production figures) plus casing-failure, well-pad and vent-flow/gas-migration
feeds, with a bulk "All-Alberta" file behind the Product and Services Catalogue. The Core Research
Centre holds the physical core/cuttings archive — separate from the web datasets, viewing/ordering
is by appointment through AER's own procedure. (Alberta Geological Survey's own open-data hub,
geology-ags-aer.opendata.arcgis.com, overlaps here for core/borehole records but is left to a
geologic-map-cell entry to avoid duplicating this one.)
