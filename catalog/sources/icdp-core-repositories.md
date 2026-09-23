---
id: icdp-core-repositories
name: ICDP Core Repositories and Sample/Data Management (mDIS)
operator: "[[International Continental Scientific Drilling Program]]"
portal: https://www.icdp-online.org/support/data-samples/
scale_tier: sample
resolution: "drill-core scale: core scans, lithological descriptions, IGSN-tagged sample records"
data_types: [core-sample, borehole]
regions: [global]
coverage:
  name: Global (discrete sites)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://www.icdp-online.org/support/data-samples/
    notes: >
      per-project drilling data/samples managed in mDIS (mobile Drilling Information System,
      GPL-2.0, LAMP stack); hierarchical program->expedition->site->hole->core->section->sample
      records with auto-assigned IGSNs; physical cores held at MARUM Bremen (cold storage) and
      German Geological Survey Berlin-Spandau (room-temp storage)
license: accessible to science team during project + moratorium; open to community post-moratorium
added: 2026-08-11
verified: 2026-08-11
---

Continental-drilling counterpart to IODP: core repositories at MARUM (Bremen, 4°C storage) and
the German Geological Survey (Berlin-Spandau, room temperature), with per-project metadata
managed through ICDP's own mDIS tool rather than a single searchable web catalog. Gotcha: mDIS
is project-scoped and largely used internally by drilling teams — there is no equivalent of
IODP's LORE for browsing core imagery across projects; discovery is mostly per-project via the
ICDP website and direct repository contact.
