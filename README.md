# SatQuery AI Lite

**Evidence-first remote sensing intelligence — lightweight, browser-first, independently deployable.**

> Don't just detect what changed. Show why the result should be trusted, where the evidence
> exists, and what could make the conclusion wrong.

Lite is deliberately **not** a smaller copy of the full SatQuery AI platform. It has one job:
make a small set of classical image-analysis capabilities *trustworthy* — every answer comes with
its diagnostics, spatial evidence, uncertainty, method, limitations and provenance.

## What it does (Evidence Engine v1)

| Capability | What you get |
|---|---|
| **Image quality intelligence** | brightness, contrast, clipping, sharpness, noise, edge density, resolution → flags, implications, and an explicit `PROCEED / PROCEED_WITH_WARNINGS / REQUEST_BETTER_IMAGERY / REJECT` decision |
| **Temporal alignment diagnostics** | translation estimate (phase correlation), correlation peak, valid overlap, NCC → `GOOD / WARNING / UNRELIABLE`. *Translation only — not geospatial registration, and every response says so.* |
| **Spatial evidence engine** | configurable grid (2×2 … 12×12); per-cell changed pixels, intensity, stability, local quality and evidence strength; concentration / distribution / cohesion analysis; clickable in the UI |
| **Uncertainty & sensitivity engine** | the same comparison at several thresholds; stable vs ambiguous change; noise-floor and global-brightness-offset checks; what-if shift compensation |
| **Capability-aware agent** | NDVI from RGB? calibrated SAR from a PNG? hectares without georeferencing? Answered with an explanation of what is missing — never a guess |
| **Geospatial Lite Bridge** | GeoTIFF metadata (CRS/EPSG, transform, pixel size, bounds, bands, NoData, COG-like layout) with no raster dependencies; STAC *item metadata* inspection and pair compatibility |
| **Reports** | JSON, Markdown, regions CSV, sensitivity CSV and a printable HTML report — all generated from the same backend result shown on screen |

The five original analysis tools (caption, VQA, grounding, change, optical–SAR fusion) are kept,
now clearly labelled as **heuristics**. There is **no model confidence anywhere**: v0's invented
confidence numbers were replaced by a stated-rule `reliability` assessment (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#reliability-instead-of-confidence)).

## What it deliberately does *not* do

- No trained vision-language model, no GPU, no fake AI confidence.
- No physical measurements (hectares, km²) or geographic coordinates of features.
- No spectral indices (NDVI/NDWI…) from RGB; no calibrated SAR processing.
- No registration/orthorectification claims from a visual translation estimate.
- No network access from the analysis path (STAC input is pasted JSON; nothing is fetched).

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest                                    # 150+ tests, no network, no fixtures on disk
SATQUERY_SERVE_STATIC=1 uvicorn api.analyze:app --reload
# open http://127.0.0.1:8000
```

Frontend end-to-end test (jsdom against the real API):

```bash
python tests/e2e/make_fixtures.py
cd tests/e2e && npm install
cd ../.. && SATQUERY_SERVE_STATIC=1 uvicorn api.analyze:app --port 8765 &
cd tests/e2e && npm test
```

## API

| Endpoint | Purpose |
|---|---|
| `GET  /api/health` | liveness + versions |
| `GET  /api/capabilities` | tasks, modes, limits, thresholds, geospatial phases, **what is not supported** |
| `POST /api/analyze` | full analysis → evidence-backed result ([schema](docs/API.md)) |
| `POST /api/diagnostics` | quality, alignment, suitability and GeoTIFF metadata only |
| `POST /api/stac/inspect` | STAC item metadata + pair compatibility (`change` / `fusion`) |

Every failure has the same shape: `{ "status", "ok": false, "error": { "code", "message" } }` — no stack traces.

## Deployment (Vercel)

Unchanged: `api/analyze.py` is the entrypoint (`pyproject.toml → [tool.vercel]`), dependencies are
`fastapi`, `numpy`, `pillow`. Limits are derived from Vercel's 4.5 MB request-body cap
(1.6 MB per image, 3.2 MB total) and can be overridden with `SATQUERY_MAX_*` environment variables.
CORS is same-origin by default; set `SATQUERY_CORS_ORIGINS` (comma-separated) to allow others.
Measured locally: a 2 × 12 MP JPEG pair is analysed in ≈ 0.7 s using ≈ 300 MB.

## Repository layout

```
api/analyze.py          Vercel entrypoint (re-exports backend.app:app)
backend/
  app.py                FastAPI routes, strict request schemas, structured errors, body-size limit
  pipeline.py           analysis orchestration (decode → gate → diagnostics → tool → reliability → report)
  ingest.py             hardened decoding (limits, formats, bit depth, hashing)
  config.py             every threshold and limit, in one place
  capabilities.py       capability registry + query gate
  diagnostics.py        image quality · alignment · suitability
  change_engine.py      Change Intelligence Engine
  evidence/             spatial_grid · region_analysis · evidence_summary
  uncertainty/          sensitivity · stability · ambiguity
  geospatial/           metadata (GeoTIFF) · validation · stac
  reliability.py        evidence level (replaces invented confidence)
  methodology.py        methods + limitations returned with every result
  report.py             Markdown / CSV / structured report from the result
  tools/                caption · vqa · grounding · change · fusion (classical heuristics)
public/index.html       Evidence workspace (renders API output only)
tests/                  unit, API, geospatial, report-parity and e2e tests
docs/                   ARCHITECTURE · API · ROADMAP
```

`agentic_controller.py`, `vqa_tool.py`, `finetune_bigearthnet.py` and `satquery-ai-prototype.html`
at the repository root are the original v0 scaffolds/prototype; they are not part of the deployed app.

## Design rules

1. The frontend never invents scientific state — it renders the backend result.
2. Every number in a sentence comes from a measurement; every sentence lists its `basis`.
3. A limitation that applies is stated every time, not buried.
4. When the data cannot support a conclusion, say so (`UNAVAILABLE_CAPABILITY`, `INSUFFICIENT_INPUT`, `UNSUPPORTED` evidence level).
