---
id: epos-multiscale-laboratories
name: EPOS Multi-Scale Laboratories (TCS-MSL)
operator: "[[Utrecht University]]"
portal: https://www.epos-eu.org/tcs/multi-scale-laboratories
scale_tier: sample
resolution: "lab-scale microscopy/microstructure and rock-physics measurements on hand samples and thin sections"
data_types: [microstructure]
regions: [europe]
coverage:
  name: Europe (network of laboratories)
  bbox: [-25.0, 34.0, 45.0, 71.0]
fetch:
  - method: manual
    url: https://www.epos-eu.org/tcs/multi-scale-laboratories/data-services
    notes: >
      MSL catalogue aggregates dataset metadata from partner repositories (chiefly GFZ Data
      Services) across four domains: analogue modelling, paleomagnetism, rock physics/HP-T,
      analytical labs (incl. electron microscopy/microstructure); no single bulk-download API,
      individual datasets are fetched from whichever repository holds them
license: varies per contributing repository/dataset
added: 2026-08-11
verified: 2026-08-11
---

Federation of over 100 European rock-physics, microscopy, paleomagnetism and analogue-modeling
labs under the EPOS umbrella, coordinated by Utrecht University. The "portal" is a metadata
catalogue, not a data lake — it points into GFZ Data Services and other partner repositories
rather than hosting imagery itself. Gotcha: the microscopy/microstructure vocabulary work
(with StraboSpot's StraboMicro tool) was still described as in development as of this check,
so microstructure coverage is thinner and less standardized than the rock-physics side.
