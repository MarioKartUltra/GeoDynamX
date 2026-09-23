---
id: iodp-core-repositories
name: IODP Core Repositories and Curation (LORE/JANUS)
operator: "[[International Ocean Discovery Program]]"
portal: https://www.iodp.org/resources/access-data-and-samples
scale_tier: sample
resolution: "drill-core scale: line-scan core images, thin sections, closeup/handheld photos, per-section sample records"
data_types: [core-sample, borehole]
regions: [global]
coverage:
  name: Global (discrete sites)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://web.iodp.tamu.edu/LORE/
    notes: LIMS Online Reports Explorer — core images/descriptions/sample data for IODP expeditions from 317 onward
  - method: manual
    url: https://iodp.tamu.edu/curation/index.html
    notes: physical core repositories (Gulf Coast, Bremen, Kochi) and sample-request procedure; Janus DB covers ODP/DSDP legs through 312
license: IODP Sample, Data, and Obligations Policy (open post-moratorium)
added: 2026-08-11
verified: 2026-08-11
---

Ocean-drilling core curation across three permanent repositories (Gulf Coast, Bremen, Kochi)
plus the LORE/JANUS databases for core images, descriptions, and physical/chemical
measurements. Gotcha: a standard 1-year post-expedition moratorium restricts sample and some
data access to the science party; the moratorium can extend longer for post-cruise sampling.
Full IODP/ODP/DSDP catalog is scattered across LORE (IODP ≥317), JANUS (ODP/DSDP legs 1–312),
and format-specific tools (Globus app for high-res linescan/thin-section imagery) rather than
one unified interface.
