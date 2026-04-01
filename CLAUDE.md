# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLPK (Scene Layer Package) texture redaction pipeline. Takes a `.slpk` 3D mesh archive and a GeoJSON file of redaction polygons, then blurs the texture regions that correspond to those polygons using a 3D voxel grid. Only textures are modified — geometry buffers are left untouched.

## Commands

```bash
# Install dependencies
uv sync

# Run the pipeline
uv run python redact.py --slpk input.slpk --polygons regions.geojson --output output.slpk

# Full options
uv run python redact.py --slpk input.slpk \
    --polygons regions.geojson \
    --output output.slpk \
    --voxel-size 2.0 \
    --blur-sigma 1.5
```

No tests, linting, or CI are configured yet.

## Architecture

The pipeline in `redact.py` runs three phases:

1. **Build voxel grid** — Iterate leaf nodes (finest LOD), extract vertices inside redaction polygon, sample texture color at each vertex's UV coordinates, accumulate into a 3D voxel grid (world space, default 2m resolution).
2. **Blur voxel grid** — Apply 3D Gaussian blur (sigma=1.5 cells = 3m) using weighted blur that avoids pulling colors from empty/unoccupied voxels.
3. **Paint all LOD levels** — For all candidate nodes (leaves + coarser LODs), identify triangles intersecting the redaction polygon, build UV mask, then for each masked pixel: compute world XYZ via barycentric interpolation in UV space, sample blurred voxel grid, write color back to texture. Includes 4px UV padding into dead space to prevent bilinear filtering artifacts.

Finally, the output SLPK is rebuilt entry-by-entry with modified textures written in place and a regenerated hash index.

Key design choices:
- **3D voxel grid** ensures seamless redaction across UV atlas seams (same world position → same color) and across LOD transitions (all LODs sample the same grid).
- **Per-pixel reverse mapping** (UV → world via barycentric in UV space) handles steep/vertical surfaces without distortion.
- **Boundary triangle clipping** — full triangles (all 3 vertices inside polygon) are rasterized directly; boundary triangles are tested per-pixel against the polygon.
- Processes **all LoD levels**, not just leaves, so zoomed-out views are also redacted.
- JPEG output at quality=100 with 4:4:4 subsampling to minimize compression artifacts.

## Known Limitations

- No draco-compressed geometry support yet.
- No KTX2/Basis supercompressed texture support yet.
- CRS is hardcoded to EPSG:3011 (SWEREF99 18 00) — needs parameterization for other regions.
- `main.py` is a placeholder; the real entry point is `redact.py`.

## Key Files

- `redact.py` — Entire pipeline implementation (~510 lines).
- `plan.md` — Original architecture plan with edge cases and future work.
- `region.geojson` — Sample redaction polygon in Stockholm for testing.
- `pyproject.toml` — Dependencies managed via `uv`; Python ≥3.12.
