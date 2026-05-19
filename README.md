# esri-slpk-redaction

Blurs regions of an Esri SLPK mesh that fall inside a geographic polygon. Textures only; geometry buffers stay untouched.

The blur source is a 3D voxel grid in world space, so the same world position resolves to the same colour regardless of which UV atlas or which LoD samples it. Atlas seams and LoD transitions stay consistent.

## Requirements

Python 3.12 or newer, [`uv`](https://docs.astral.sh/uv/), an SLPK with JPEG textures and uncompressed geometry buffers. Draco and KTX2 are not supported yet.

## Install

```bash
uv sync
```

## Usage

```bash
uv run python redact.py \
    --slpk input.slpk \
    --polygons regions.geojson \
    --output output.slpk \
    --crs EPSG:3011
```

`--polygons` is a GeoJSON file in EPSG:4326 (FeatureCollection, single Feature, or bare Geometry). All features are unioned into a single redaction mask. Polygons, MultiPolygons, and holes are all supported. Features outside the mesh are silently ignored. `--crs` is the projected CRS the SLPK's vertex positions live in; default is EPSG:3011 (SWEREF99 18 00).

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--slpk` | required | Input SLPK |
| `--polygons` | required | GeoJSON polygon(s), EPSG:4326 |
| `--output` | required | Output SLPK |
| `--crs` | `EPSG:3011` | CRS of SLPK vertex coordinates |
| `--voxel-size` | `2.0` | Voxel edge length, metres |
| `--blur-sigma` | `1.5` | Gaussian sigma, voxel cells |

## How it works

Three phases.

1. **Sample.** Walk the leaf nodes, find vertices inside the polygon, read the texture colour at each vertex's UV, accumulate into a 3D grid in world space (default 2 m cells).
2. **Blur.** 3D Gaussian across the grid, weighted so empty cells don't pull the colour toward zero.
3. **Paint.** For every candidate node (leaves and interior LoDs), find triangles that intersect the polygon, rasterise a UV mask, and for each masked texel resolve world XYZ via barycentric interpolation and look up the blurred grid. A 4 px UV dilation into dead space stops bilinear filtering from bleeding the original texture back across triangle edges.

Repacking writes modified JPEGs in place, leaves every other entry byte-for-byte, and regenerates the v1.7+ hash index (`@3dtilesIndex1@`).

## Limitations

- No Draco-compressed geometry.
- No KTX2/Basis-compressed textures.
- Each polygon gets its own voxel grid; dispersed polygons scale linearly, not by bounding-box area.
- No tests, no CI.

## License

MIT. See [LICENSE](LICENSE).
