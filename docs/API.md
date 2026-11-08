# API reference (schema 2.0)

## `POST /api/analyze`

```jsonc
{
  "mode": "single | bitemporal | cross",
  "query": "string, 1-500 chars",
  "images": ["data:image/png;base64,…"],          // 1 (single) or 2 (pairs); PNG, JPEG, 8-bit TIFF/GeoTIFF
  "options": {                                     // change analysis only
    "grid_size": 4,                                // 2-12
    "base_threshold": 28,                          // 1-254
    "thresholds": [20, 28, 40]                     // 2-7 distinct, base is always included
  },
  "client_notes": [{ "index": 0, "resized_from": [w,h], "resized_to": [w,h] }]   // optional, echoed into provenance
}
```

### Status values

| `status` | `ok` | Meaning |
|---|---|---|
| `COMPLETED` | true | analysis ran; no diagnostic raised a concern |
| `COMPLETED_WITH_WARNINGS` | true | analysis ran; see `reliability`, `diagnostics.suitability` |
| `UNAVAILABLE_CAPABILITY` | false | the question needs data/capability this input or deployment lacks (`unmet_requirements`, `supported_alternatives`) |
| `INSUFFICIENT_INPUT` | false | e.g. change question with one image; wrong image count |
| `REJECTED_INPUT` | false | invalid request/image/limits (`error.code`) |
| `ERROR` | false | unexpected failure; no internals exposed |

### Successful response (abridged)

```jsonc
{
  "schema_version": "2.0", "engine_version": "1.0.0",
  "request_id": "uuid", "analysis_id": "sha256 of engine+inputs+query+parameters", "created_at": "…",
  "status": "COMPLETED_WITH_WARNINGS", "ok": true, "task": "change-describe", "tool": "…",
  "answer": "generated from measurements, ends with the reliability caveat",
  "confidence": null,
  "reliability": { "level": "CAUTION", "factors": [{ "name", "status": "OK|CONCERN|SEVERE", "detail", "value" }], "concerns": [], "rule": "…" },
  "diagnostics": {
    "quality": [{ "overall", "metrics": {…}, "levels": {…}, "flags": [{ "code", "severity", "message", "implication" }], "analysis_implications": [] }],
    "alignment": { "state", "dx_px", "dy_px", "peak_correlation", "valid_overlap_fraction", "ncc_aligned", "reasons": [], "limitation": "…" },
    "suitability": { "decision", "reasons": [], "rule": "…" },
    "geospatial_pair": { "state", "findings": [] }
  },
  "evidence": {                                    // change tasks
    "measurements": {…}, "spatial_evidence": { "grid_size", "cells": [ { "region_id": "grid_5", "bbox_px", "bbox_norm", "changed_pixel_count", "analysed_pixel_count", "change_percentage", "retention_at_strictest", "evidence_strength", "quality_warning", "interpretation", … } ] },
    "concentration": {…}, "sensitivity": { "thresholds": […], "sensitivity_level" }, "stability": {…},
    "noise_floor": {…}, "radiometric": {…}, "alignment_impact": {…},
    "statements": [{ "id", "text", "basis": {…} }]
  },
  "layers": [{ "id", "label", "description", "width", "height", "data": "data:image/png;base64,…" }],
  "methodology": { "name", "steps": [], "parameters": {} }, "limitations": [],
  "inputs": [{ "role", "sha256", "bytes", "format", "original_size", "working_size", "georeferenced", "geospatial": {…}|null, "warnings": [] }],
  "provenance": { "inputs": […], "working_grid": {…}, "parameters": {…}, "processing_steps": [] },
  "report": { "sections": […], "markdown": "…", "regions_csv": "…", "sensitivity_csv": "…" },
  "trace": [{ "time", "label", "detail" }], "timings_ms": {…}
}
```

Layers for change analysis: `before`, `after`, `difference`, `change_mask`, `evidence_grid`, `uncertain`, `combined`.
Single-image tasks return `original` (+ `overlay`); fusion returns `optical`, `sar`, `fused`.

### Error codes

`INVALID_REQUEST`, `INVALID_OPTIONS`, `WRONG_IMAGE_COUNT`, `EMPTY_IMAGE`, `INVALID_BASE64`, `IMAGE_TOO_LARGE` (413),
`IMAGES_TOO_LARGE` (413), `IMAGE_TOO_MANY_PIXELS` (413), `PAYLOAD_TOO_LARGE` (413), `IMAGE_TOO_SMALL`, `UNSUPPORTED_FORMAT` (415),
`UNSUPPORTED_BIT_DEPTH` (415), `UNSUPPORTED_RASTER` (415, includes `geospatial` metadata), `CORRUPT_IMAGE`, `UNUSABLE_IMAGE` (422),
`TOOL_FAILURE` (500), `INTERNAL_ERROR` (500), plus the capability codes `SPECTRAL_BANDS_UNAVAILABLE`,
`CALIBRATED_SAR_UNAVAILABLE`, `PHYSICAL_AREA_UNAVAILABLE`, `GEOCOORDINATES_UNAVAILABLE`, `ACQUISITION_METADATA_UNAVAILABLE`,
`OBJECT_COUNTING_UNAVAILABLE`, `CHANGE_NEEDS_TWO_OBSERVATIONS`, `FUSION_NEEDS_OPTICAL_AND_SAR`.

## `POST /api/diagnostics`

`{ "images": [one or two] }` → `diagnostics`, `inputs` (including `geospatial` metadata), `working_grid`. No tool is run.

## `POST /api/stac/inspect`

`{ "items": [STAC Item, …1-2], "purpose": "change" | "fusion" }` (`purpose` required for two items) →
`items[]` (collection, platform, datetime, bbox, cloud cover, EPSG, modality + basis, bands, assets with what Lite v1 can do)
and `comparison` (`COMPATIBLE | COMPATIBLE_WITH_WARNINGS | INCOMPATIBLE`, time gap, bbox IoU, findings).
Metadata only; nothing is fetched.
