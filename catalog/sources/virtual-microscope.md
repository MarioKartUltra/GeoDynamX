---
id: virtual-microscope
name: Virtual Microscope
operator: "[[The Open University]]"
portal: https://virtualmicroscope.org/
scale_tier: sample
resolution: "petrographic thin sections (~30 micron slices) under polarizing-microscope magnification"
data_types: [microstructure]
regions: [global]
coverage:
  name: Global — museum and university thin-section/sample collections
  bbox: [-180.0, -90.0, 180.0, 90.0]
fetch:
  - method: manual
    url: https://virtualmicroscope.org/
    notes: >-
      OER partnership between the Open University and museums/universities/planetary-material
      archivists. Confirmed live. Holdings include a JISC-funded UK rock thin-section set
      (100+), Darwin's Beagle-voyage specimens, and NASA-collected meteorite/lunar samples,
      browsed as interactive thin-section/specimen viewers rather than bulk downloads.
formats: [jpg]
license: CC BY-NC-SA 2.0 (third-party materials may carry different terms)
added: 2026-08-11
verified: 2026-08-11
---

A teaching-oriented thin-section/microscopy archive rather than a research image database —
useful as a microstructure sample source but the interface is built around interactive
in-browser viewing, not batch export, and no bulk-download or API path is documented on the
homepage. Non-commercial licence (NC clause) is a real constraint for any downstream reuse
beyond education.
