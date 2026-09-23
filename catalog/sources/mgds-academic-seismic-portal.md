---
id: mgds-academic-seismic-portal
name: MGDS — Marine Geoscience Data System (Academic Seismic Portal)
operator: "[[Lamont-Doherty Earth Observatory]]"
portal: https://www.marine-geo.org/
scale_tier: global
resolution: "survey-level: multi-channel, single-channel, chirp sub-bottom and sonobuoy seismic; raw swath bathymetry (e.g. EM122/EM302); opportunistic coverage wherever US academic vessels have surveyed, not continuous"
data_types: [seismic-reflection, bathymetry]
regions: [global]
coverage:
  name: Global oceans (US academic research-vessel coverage)
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: http://www.marine-geo.org/services/search/datasets
    notes: Dataset search API; sibling `/services/search/entries` for finer-grained entries.
  - method: api
    url: http://www.marine-geo.org/services/seismicxml/xml?id={dataset_id}
    notes: "Seismic-survey metadata as XML, e.g. ?id=EW0207. Documented via /tools/fileserverinfo.php, /tools/filedownloadserverinfo.php, /tools/archivedownloadserverinfo.php (FileServer/FileDownloadServer/ArchiveDownloadServer WADLs)."
  - method: manual
    url: https://www.marine-geo.org/collections/#!/collection/Seismic
    notes: Academic Seismic Portal map/collection browser (2D/3D MCS, single-channel, chirp, sonobuoy); GMRT compiled bathymetry grid is a separate MGDS-hosted product already catalogued under gmrt.md.
formats: [segy, netcdf, xml]
license: CC BY-NC-SA 3.0 (site-wide, per marine-geo.org/about/legal.php)
added: 2026-08-11
verified: 2026-08-11
---

NSF-funded repository (hosted at LDEO) for US academic marine geophysics: raw multi-channel
and single-channel seismic reflection, chirp sub-bottom, sonobuoy, and raw swath bathymetry
from research-vessel cruises — distinct from the GMRT synthesis grid it also hosts (already
in this catalog). Search/metadata APIs are open and scriptable; actual seismic/bathymetry
files download via FileDownloadServer/ArchiveDownloadServer per dataset ID, not a single bbox
clip. Coverage is wherever a funded academic cruise has gone, so it is dense along some
margins and empty elsewhere — check the collection map before assuming coverage.
