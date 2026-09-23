---
id: sesar-sample-registry
name: SESAR — System for Earth and Extraterrestrial Sample Registration
operator: "[[Lamont-Doherty Earth Observatory]]"
portal: https://www.geosamples.org/
scale_tier: sample
resolution: "per-specimen registry records (hand sample to core/thin-section, IGSN-identified)"
data_types: [core-sample]
regions: [global]
coverage:
  name: Global (discrete sites)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://geosamples.github.io/sesar-doc/
    notes: REST API for registering/querying IGSN sample metadata records
  - method: manual
    url: https://www.geosamples.org/
    notes: web catalog search over registered samples
license: CC BY-NC-SA 3.0
added: 2026-08-11
verified: 2026-08-11
---

Global sample registry, not a data archive: assigns International Geo Sample Numbers (IGSN,
DOI prefix 10.58052) to physical specimens so they can be cited and tracked across
institutions. Gotcha: records are metadata about samples (collection locality, material type,
custodian) — the physical sample and any associated imagery/analyses live elsewhere (often
cross-referenced from EarthChem, IODP/ICDP, or an institutional repository). Part of the NSF-
funded IEDA2 facility, hosted at Columbia's Lamont-Doherty Earth Observatory.
