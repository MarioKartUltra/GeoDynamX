# DynamiX data-source catalog

A scale-organized registry of portals hosting structural-geological and geophysical data.
The notes in `sources/` ARE the registry — frontmatter is the record, body is free prose.

- [[By Tier]] · [[By Region]] · [[By Data Type]]

Rebuild indexes + `_generated/catalog.json` after editing sources:

    python catalog/tools/build.py

Rules: never store bulk data here (metadata, footprints, fetch recipes only);
every `portal`/`fetch` URL is live-checked before an entry lands (`verified` stamps it).
