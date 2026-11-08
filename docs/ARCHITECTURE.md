# Architecture

## Request flow

```
POST /api/analyze
  │  strict schema validation (pydantic) · body-size limit (pure ASGI, also bounds chunked uploads)
  ▼
pipeline.run_analysis
  1  input count vs mode
  2  task classification (rule-based, auditable)
  3  ingest.decode_image  ── base64 strict · size · pixel count (from header) · format allow-list ·
  │                          bit depth · frames · EXIF · alpha · SHA-256 · GeoTIFF metadata
  4  capabilities.check_query ── UNAVAILABLE_CAPABILITY / INSUFFICIENT_INPUT stop here, no tool runs
  5  working grid (single: ≤380 px; pair: A's grid ≤384 px, B resampled onto it)
  6  diagnostics: quality ×N · alignment · aspect · geospatial pair · suitability
  │     REJECT stops here
  7  tool / engine (change_engine for temporal tasks)
  8  reliability.assess_reliability
  9  provenance (hashes, parameters, steps) · methodology · limitations · report
  ▼
{ status, answer, reliability, diagnostics, evidence, layers, methodology, limitations, provenance, trace, report }
```

## Change Intelligence Engine

Input: two `(H, W, 3)` uint8 arrays on the same grid. `gray = mean(R, G, B)` (the definition v0's
threshold was tuned against).

1. `diff = |gray_B − gray_A|`; changed = `diff > base_threshold`.
2. **Sensitivity** — repeat at thresholds `T`; `sensitivity_index = (max − min) / mean` of changed %.
3. **Stability** — `stable = diff > max(T)` (changed at *every* threshold), `ambiguous = min(T) < diff ≤ max(T)`.
   Stability is robustness to the threshold, **not** evidence of real change.
4. **Spatial grid** — per cell: counts, %, mean/p95 difference, stable/ambiguous pixels, retention,
   local clipping/contrast/sharpness, quality warning, evidence strength
   (`NONE < min%`; `STRONG` ≥ strong% and retention ≥ 0.70; `MODERATE` retention ≥ 0.40; else `WEAK`; a quality
   warning demotes one step).
5. **Concentration** — top-cell share, entropy, Gini, *cohesion* (share of changed pixels with ≥ 3 changed
   neighbours, separates compact regions from salt-and-pepper noise), centroid phrase.
6. **Checks** — noise floor (`threshold / σ_diff ≥ 3`), global brightness offset (`|median(B−A)| ≥ 8`),
   what-if shift-compensated change %.
7. **Statements** — generated from the numbers; each carries a `basis` object.

### Alignment diagnostic

Windowed phase correlation (sub-pixel via parabolic peak fit) → `(dx, dy)`, correlation peak, overlap after
shift, NCC before/after. Convention: content moved from A to B by `(dx, dy)`; `+dx` right, `+dy` down.
`UNRELIABLE` for ≥ 4 px shift, < 85 % overlap, no coherent peak, untextured input, or aspect-ratio mismatch ≥ 10 %.
A weak peak is `WARNING` (it can also mean extensive genuine change), not proof of misalignment.
**Not** orthorectification, terrain correction, rotation/scale estimation or registration.

## Reliability instead of confidence

v0 produced a `confidence` from arbitrary formulas. A heuristic pipeline has no calibrated probability, so none is
reported (`confidence: null` always, with an explanation). Instead:

| Level | Rule |
|---|---|
| `SUPPORTED` | no concerns raised by any implemented diagnostic |
| `CAUTION` | exactly one concern |
| `WEAK` | two or more concerns |
| `UNSUPPORTED` | any severe concern (e.g. unreliable alignment) — the answer says it must not be relied on |

`SUPPORTED` means *no diagnostic found a reason for doubt*, not *correct*. Colour-rule / proxy tools carry a permanent
`method` concern, so they can never read `SUPPORTED`.

## Geospatial Lite Bridge

* **Phase A (ready)** — `geospatial/metadata.py`: bounds-checked TIFF/BigTIFF tag reader, GeoKeys, tiepoint+scale and
  affine transforms, PixelIsPoint handling, NoData, band descriptions, COG-like layout heuristic. WGS84 bounds only for
  EPSG:4326/3857 (no PROJ). Never decodes pixels, never executes file content, caps tag sizes and IFD count.
* **Phase B (ready)** — `geospatial/stac.py`: inspects STAC Item JSON supplied by the caller; classifies assets by what
  Lite can do with them; compares pairs for `change` or `fusion`. **Nothing is fetched.**
* **Phase C (not implemented)** — native raster processing. Requires Rasterio/GDAL; Vercel runtime/size constraints must
  be evaluated first. A GeoTIFF is never silently converted to a JPEG and called geospatial analysis: 8-bit GeoTIFFs are
  analysed in image space and the provenance says so; anything else returns its metadata with `UNSUPPORTED_RASTER`.

## Security posture

Limits before expensive work (bytes, megapixels from the header, dimensions, frames, bit depth, formats); strict base64;
MIME prefix not trusted (format sniffed); bounded TIFF parser; no network from the analysis path; no `eval`/pickle;
structured errors without internals; CSV formula-injection guard; same-origin CORS by default; all API text rendered
with `textContent` (no `innerHTML` with data).

## Testing

`pytest`: synthetic imagery only (deterministic, in-memory). Covers every case in the validation table — identical
images, uniform brightness change, translations of known size, blur, overexposure, noise, local change in a known cell,
threshold variation vs brute force, stable/ambiguous regions, different dimensions, invalid/oversized/malformed input,
missing geospatial metadata — plus capability gating, determinism/replay, report ↔ evidence parity, GeoTIFF metadata
against files built to known values, STAC rules, and equivalence of the vectorised legacy tools with their original loops.
`tests/e2e`: jsdom drives the real page against the real API.
