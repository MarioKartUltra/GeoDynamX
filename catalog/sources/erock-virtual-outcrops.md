---
id: erock-virtual-outcrops
name: eRock — Open-Access Repository of Virtual Outcrops and Samples
operator: "[[University of Aberdeen]]"
portal: https://www.e-rock.co.uk/
scale_tier: sample
resolution: "photogrammetric 3D virtual-outcrop and hand-sample models (sub-cm to decimeter mesh detail, model-dependent)"
data_types: [outcrop-model]
regions: [global]
coverage:
  name: Global — scattered virtual-outcrop and sample localities
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://www.e-rock.co.uk/
    notes: >-
      ~40 web-viewable 3D models (Sketchfab-embedded), browsable by geological theme
      (carbonates, clastics, crystalline, folding/fracturing, faults, unconformities) or by
      region (Pembrokeshire, NW Highlands, NE Scotland, etc.) with a map interface. Most
      models are downloadable under the model's own CC BY licence; contributors may opt a
      model out of downloading.
formats: [obj, gltf]
license: CC BY (per-model; some models set non-downloadable)
added: 2026-08-11
verified: 2026-08-11
---

Aberdeen-led (Cawood & Bond) open virtual-outcrop repository aimed at teaching and outreach
as much as research — folds, faults and unconformities as photogrammetric 3D models with
field photos, maps and cross-sections attached. Coverage is a scattered handful of curated
localities, not a systematic survey, and per-model download permission varies, so check each
model's own licence flag before assuming it can be pulled.
