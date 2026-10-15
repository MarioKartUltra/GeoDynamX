# The device contract

Written by `docs/reference/make_tables.py` from the device registry; edit a declaration,
then run it. A device is one Python class; nothing here needs a base class.

## What a device declares

| attribute | meaning |
|---|---|
| `name` | the id: lowercase, digits, underscores; stable, it names the step in a saved chain, a cache key and a `.gprv` step |
| `label`, `families` | display name and the families it is listed under (`Labs` for an experimental device, `()` to leave the palette) |
| `params` | a tuple of `Param`; `validate_params` fills defaults and rejects unknown keys |
| `compute(field, params, progress=None)` + `cache_key(params)` | a Transform; `apply(extrema, params)` makes a Filter |
| `consumes`, `produces` | alternatives of contract names the step needs, and the set it adds (`<received>` passes the input on) |
| `capabilities` | a frozenset of the names below; absent, the engine probes the attributes instead |
| `role = "primary"`, `chain_producer`, `chain_consumer`, `superseded_by`, `shell_placed`, `band_variant_of` | the roles the shell derives its device sets from |
| `outputs` | `Output` declarations the layer's rows show, computed by `compute_output` |

A device file in the user's devices folder exposes `DEVICES = [...]`; a package registers through
the `geodynamix.devices` entry-point group. `python -m dynamix.new_device <id>` writes a file with a
golden test. A file that fails to import or validate is a greyed browser row with its error.

## Capabilities

| capability | probed attributes |
|---|---|
| `FieldStage` | `field_stage` |
| `SourceReader` | `reads_source` |
| `VectorSource` | `vector_source` |
| `RegionAware` | `roi_margin`, `compute_roi` |
| `Cancellable` | `wants_cancel` |
| `Previewable` | `preview` |
| `Viewable` | `view` |
| `OutputProducer` | `outputs` |
| `SelectionAware` | `selection_aware` |
| `SelfReporting` | `reading`, `data_hints` |
| `StripExtras` | `derived_reading`, `param_groups`, `check` |

## Contracts

| name | meaning |
|---|---|
| `Field` | a single-band field |
| `Stack` | a band stack |
| `Points` | a point set |
| `ExtremaStack` | an extrema stack |
| `PointAlpha` | a per-point α |
| `VChains` | V-chains |
| `Topology` | the H/V topology |
| `HMap` | an h-map |
| `Components` | decomposition components |

## Families

`Data preparation`, `Edge detection`, `Regularity`, `Skeletons`, `Decomposition`, `Reconstruction`, `Topology`, `Selection: extrema`, `Selection: V-chains`, `Points`, `Labs`, `Other`

## Roles today

- primary analyzers: `wtmm2d`, `mz_edges`, `wavelet_skeleton`, `cdf_edges`, `pm_edges`, `holder_map`, `holder_measure`, `holder_multiaffine`, `band_recon`, `band_recon_measure`, `band_recon_multiaffine`
- chain producers: `wtmm2d`, `wtmm2d_roi`, `mz_edges`, `wavelet_skeleton`, `cdf_edges`, `pm_edges`
- superseded: `holder_map` by `holder_measure`, `band_recon` by `band_recon_measure`, `band_recon_measure` by `holder_measure`, `band_recon_multiaffine` by `holder_multiaffine`, `tucker_havok` by `tucker_HOOI_HOSVD`, `scale_select` by the Transport
- shell-placed: `bus`, `group_paint`, `derived_vectors`
