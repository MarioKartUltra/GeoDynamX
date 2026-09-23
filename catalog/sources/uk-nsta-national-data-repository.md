---
id: uk-nsta-national-data-repository
name: UK National Data Repository (NDR) — wells and seismic
operator: "[[North Sea Transition Authority]]"
portal: https://ndr.nstauthority.co.uk/
scale_tier: national
resolution: "per-well records; 2D/3D SEG-Y seismic volumes"
data_types: [seismic-reflection, borehole]
regions: [europe, united-kingdom]
coverage:
  name: UK Continental Shelf
  bbox: [-9.5, 49.5, 3.5, 61.5]
fetch:
  - method: manual
    url: https://ndr.nstauthority.co.uk/
    notes: "Data Discovery map + project-table interface; projects typed Well/Seis/Mhaz/Rems/Inpt under CCCCYYYYtypeNNNN IDs"
  - method: api
    url: https://www.uk-ndr.co.uk/spapidoc
    notes: "Microsoft Graph API for programmatic Project ID / File ID metadata access"
  - method: manual
    url: https://ndr.nstauthority.co.uk/newcompanyform
    auth: registration
    notes: "New Company Request Form — download access requires an org account via a company administrator"
license: "Governed by the NDR User Agreement and Terms of Sale; downloads count against per-company monthly quotas, up to 24hr package prep, physical media delivery also offered"
added: 2026-08-11
verified: 2026-08-11
---

The NSTA's National Data Repository — offshore UK Continental Shelf well and seismic licence-area
data, searchable via an interactive map/project-table (quadrant, hexbin and polygon filters) and a
Graph API for metadata. Registration-walled: organisations need a company account (existing admin
or a New Company Request Form) before downloads are enabled, and downloads are metered against
monthly quotas. Coverage is offshore-only despite the "national" scope — onshore UK hydrocarbon
licensing sits outside NSTA/NDR.
