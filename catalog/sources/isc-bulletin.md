---
id: isc-bulletin
name: ISC Bulletin — Reviewed Seismicity Catalog
operator: "[[International Seismological Centre]]"
portal: https://www.isc.ac.uk/iscbulletin/search/
scale_tier: global
resolution: "event catalog, 1900-present, all magnitudes reported by contributing networks (finer completeness than ComCat pre-1960s)"
data_types: [earthquake-catalog]
regions: [global]
coverage:
  name: Global
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: http://www.isc.ac.uk/cgi-bin/web-db-run
    notes: >-
      Live-tested (COLLECTED bulletin, CATCSV output, 2020-01-01 to
      2020-01-02, min_mag=6) — returned 4 real events in ISC's CATCSV text
      format. Query params (region/time/magnitude/depth/phases) documented at
      https://www.isc.ac.uk/iscbulletin/search/webservices/. QuakeML, CSV, ISF
      and IMS1.0 outputs also available via related endpoints.
  - method: manual
    url: https://www.isc.ac.uk/iscbulletin/search/webservices/
    notes: web-services documentation/URL builder
formats: [csv, quakeml, isf]
native_metadata:
  - format: html
    url: https://www.isc.ac.uk/citations/
license: free for scientific/educational use with citation; DOI 10.31905/D808B830
added: 2026-08-11
verified: 2026-08-11
---

Merged bulletin from 130+ contributing seismological agencies, reviewed by ISC analysts —
the deepest historical/most complete global catalog (back to 1900), at the cost of a
CGI-era query interface rather than a modern REST API. Good complement to ComCat for
pre-instrumental-era or small-magnitude events outside well-networked regions.
