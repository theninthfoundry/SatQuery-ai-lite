# Roadmap

## Done — Evidence Engine v1
Backend diagnostics & input hardening · temporal alignment diagnostics · spatial evidence engine · threshold
sensitivity & ambiguity · evidence-backed response schema · interactive workspace · JSON/CSV/Markdown/printable reports ·
GeoTIFF metadata (Phase A) · STAC item metadata (Phase B) · automated tests + CI.

## Next
1. **Phase C — raster processing.** Evaluate Rasterio/GDAL on the Vercel runtime (size, cold start). If it does not fit,
   run raster work behind a separate worker and keep Lite as the metadata/diagnostics front end. Requirements: read
   16-bit and multiband GeoTIFF/COG without silently rescaling; preserve CRS/transform through every step; windowed reads.
2. **Real change evidence beyond intensity.** Per-band differencing, radiometric harmonisation (histogram matching,
   reported as a diagnostic, never applied silently), connected-component regions with polygons.
3. **Registration.** Rotation/scale-aware estimation (log-polar / feature based) with residual reporting; still labelled
   as visual registration unless geospatial metadata supports more.
4. **Calibration of thresholds.** Replace the operating points in `backend/config.py` with values derived from labelled
   data; report measured precision/recall per scene type.
5. **Golden-scene suite.** Real (public) scene pairs with expected regions and tolerances, run in CI next to the
   synthetic tests; adversarial inputs (rotated, cloud-filled, wrong CRS, corrupt GeoTIFF).
6. **Operational.** Rate limiting at the edge, request tracing, size-aware client-side downscaling presets.

## Out of scope for Lite (belongs to full SatQuery AI)
Trained vision-language models, GPU inference, calibrated SAR processing, spectral-index engines, ground-area
measurement, object detection/counting, STAC network search.
