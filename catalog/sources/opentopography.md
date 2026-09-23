---
id: opentopography
name: OpenTopography
operator: "[[San Diego Supercomputer Center]]"
portal: https://opentopography.org/
scale_tier: local
resolution: "variable — global 30-90 m DEM mosaics (SRTM, Copernicus GLO-30/90, ALOS, NASADEM) down to sub-meter lidar/SfM point clouds at individual sites"
data_types: [lidar, dem]
regions: [global]
coverage:
  name: Global — continental DEM mosaics plus a discrete-site lidar point-cloud catalog
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: api
    url: https://portal.opentopography.org/API/globaldem
    auth: api-key
    notes: >-
      Confirmed live: an unauthenticated request returns "Error: API Key required for
      access" rather than a connection failure. Params: demtype, south/north/east/west,
      outputFormat. Free registration for the key. Separate point-cloud API covers the
      hundreds of catalogued lidar/SfM datasets (500,000+ km² combined).
formats: [geotiff, laz, las]
license: mostly CC0/public-domain source data; some contributed lidar datasets carry their own attribution terms
added: 2026-08-11
verified: 2026-08-11
---

The community high-resolution topography clearinghouse — NSF-funded, hosts both global DEM
mosaics and a large discrete-site catalog of airborne/UAS lidar and SfM point clouds (the
"local" coverage this catalog is short on). Point-and-click portal plus a REST API; API key
is free but mandatory for every endpoint, including the global-DEM one. An "OT+" paid tier
exists for higher-volume non-academic use — academic/API-key access to the core catalog
itself is free.
