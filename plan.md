# SLPK Texture Redaction — Implementation Plan

## Overview

Given: an `.slpk` (Integrated Mesh scene layer) and a set of geographic redaction polygons (e.g., GeoJSON in EPSG:4326), blur or paint over the texture regions corresponding to those polygons across all LoD levels.

The pipeline has six stages: **unpack → index → select → redact → validate → repack**.

---

## Stage 0: Unpack the SLPK

An `.slpk` is a ZIP archive (no compression on individual entries for v1.7+, or gzip-compressed entries for v1.6). The internal layout follows:

```
/
├── 3dSceneLayer.json.gz          # layer metadata (CRS, store config, geometry schema)
├── nodePages/
│   ├── 0.json.gz                 # paged node index (batches of ~64 nodes)
│   └── ...
├── nodes/
│   ├── 0/                        # root node (typically no geometry)
│   ├── 1/
│   │   ├── geometries/0.bin.gz   # uncompressed geometry buffer
│   │   ├── geometries/1.bin.gz   # draco-compressed geometry buffer (if present)
│   │   ├── textures/0.jpg        # texture atlas (JPEG, PNG, KTX2, or DDS)
│   │   ├── textures/0_0_1.bin    # compressed texture (Basis/KTX2, if present)
│   │   ├── shared/               # sharedResource descriptor (v1.6 compat)
│   │   └── ...
│   └── ...
├── metadata.json                 # SLPK version, archive hash
└── @specialIndexFileHASH128      # hash-based lookup table (v1.7+)
```

**Action:**
1. Unzip the `.slpk` into a working directory. Use Python's `zipfile` module.
2. Decompress any `.gz` entries (node pages, geometry buffers, layer JSON).
3. Parse `3dSceneLayer.json` to extract:
   - `spatialReference` → the CRS used for vertex positions and bounding volumes.
   - `store.defaultGeometrySchema` or `geometryDefinitions` → tells you the binary layout of geometry buffers (attribute order, types, whether draco is used).
   - `store.textureEncoding` → JPEG, PNG, KTX2, etc.
   - `heightModelInfo` → orthometric vs. ellipsoidal; matters for polygon intersection if your redaction polygons have a height component.

**Key detail:** Vertex positions in each node are stored as **offsets from the node's bounding volume center** (OBB center or MBS center). You must add the node's origin to get global coordinates.

---

## Stage 1: Build a Spatial Index of Nodes

Parse all node pages (`nodePages/*.json`) to build a flat list of nodes with their bounding volumes.

Each node page entry contains:
- `obb` (Oriented Bounding Box): center `[x, y, z]`, halfSize `[sx, sy, sz]`, quaternion `[qx, qy, qz, qw]`
- or `mbs` (Minimum Bounding Sphere): center `[x, y, z, radius]`
- `lodThreshold` — the LoD metric
- `resourceId` — pointer to geometry/texture resources under `nodes/`
- `parentIndex`, `firstChild`, `childCount` — tree structure

**Action:**
1. Load all node pages into a single list.
2. For each node, convert its OBB/MBS to an axis-aligned bounding box (AABB) in the layer's CRS.
3. Build an R-tree (e.g., `rtree` Python package) over these AABBs for fast spatial query.
4. Store the mapping: `node_index → resourceId` (since `resourceId` may differ from `node_index`).

**CRS note:** If the layer is in global mode (WGS84, EPSG:4326), bounding volume centers are `[longitude, latitude, elevation]`. If in a local/projected CRS, they're in that CRS's units. Your redaction polygons must be in the same CRS (reproject with `pyproj` if needed).

---

## Stage 2: Select Candidate Nodes

Query the R-tree with the bounding box of each redaction polygon to get candidate nodes.

**Action:**
1. Compute the 2D bounding box of each redaction polygon.
2. Query the R-tree → candidate node set.
3. This is a coarse filter. Actual triangle-level intersection happens in Stage 3.

**LoD consideration:** The node tree is a LoD hierarchy. A redaction polygon will typically intersect nodes at *every* LoD level — coarse parent nodes and fine leaf nodes. You must process **all** matching nodes, not just leaves, because clients may display any LoD level depending on camera distance. If you only redact leaf nodes, the un-redacted coarse parent will be visible when zoomed out.

---

## Stage 3: Parse Geometry and Identify Redaction Triangles

For each candidate node, read the geometry buffer and identify which triangles fall inside the redaction polygon(s).

### 3a. Read the geometry buffer

The binary layout (for uncompressed buffers, `geometries/0.bin`) follows a fixed attribute order:

```
[header: vertexCount(UInt32), featureCount(UInt32)]  // 8 bytes offset
[position: Float32 × 3 × vertexCount]                // xyz, relative to OBB center
[normal:   Float32 × 3 × vertexCount]                // if present
[uv0:      Float32 × 2 × vertexCount]                // texture coordinates
[uv1:      Float32 × 2 × vertexCount]                // if present (rare)
[color:    UInt8   × 4 × vertexCount]                 // RGBA, if present
[uvRegion: UInt16  × 4 × vertexCount]                 // if present (atlas sub-regions)
[featureId: UInt64 × featureCount]                    // per-feature
[faceRange: UInt32 × 2 × featureCount]                // per-feature triangle ranges
```

Which attributes are present is declared in `geometryDefinitions[0].geometryBuffers[0]` in the layer JSON. Parse accordingly — skip absent attributes.

**For draco-compressed buffers** (`geometries/1.bin`): decode with `DracoPy` or the `draco` CLI decoder. This gives you the same vertex attributes (position, uv0, color, etc.) as numpy arrays.

**Action:**
1. Read the header → `vertexCount`, `featureCount`.
2. Parse positions as `np.frombuffer(..., dtype=np.float32).reshape(-1, 3)`.
3. Parse uv0 as `np.frombuffer(..., dtype=np.float32).reshape(-1, 2)`.
4. If `uvRegion` is present, parse it — these define sub-regions within the texture atlas for repeated textures.
5. Geometry is non-indexed triangle soup: every 3 consecutive vertices form one triangle. Triangle `i` uses vertices `[3i, 3i+1, 3i+2]`.

### 3b. Transform positions to global CRS

Positions are relative to the node's OBB/MBS center. Add the center:

```python
global_positions = local_positions + node_obb_center  # broadcast [N,3] + [3]
```

For WGS84 layers: positions are `[lon_offset, lat_offset, z_offset]` in degrees and meters respectively. The center is `[lon, lat, z]`. After adding, you have `[longitude, latitude, elevation]` per vertex.

### 3c. Test triangle-polygon intersection

For each triangle (every 3 consecutive vertices), test whether it intersects the redaction polygon.

**Approach (2D, ignoring elevation):**
1. Project triangle vertices to 2D: `(longitude, latitude)` or `(x, y)` in the layer CRS.
2. Construct a Shapely `Polygon` from the three 2D vertices.
3. Test intersection with the redaction polygon(s) using `triangle_poly.intersects(redaction_poly)`.
4. Collect the indices of all intersecting triangles.

**Performance:** For large nodes (100k+ triangles), vectorize with `shapely.STRtree` or use `matplotlib.path.Path.contains_points` on triangle centroids as a fast pre-filter, then do exact intersection on the remaining candidates.

### 3d. Map triangles to UV-space pixel regions

For each intersecting triangle `i`, get its three UV coordinates:

```python
uv_a = uv0[3*i]      # [u, v], each in [0, 1] (or beyond for tiled textures)
uv_b = uv0[3*i + 1]
uv_c = uv0[3*i + 2]
```

If `uvRegion` is present, the actual texture coordinates are wrapped within the atlas sub-region:
```python
# uvRegion is [u_min, v_min, u_max, v_max] as normalized UInt16 → [0, 1]
region = uvRegion[3*i] / 65535.0  # same region for all 3 vertices of the triangle
u_actual = frac(uv) * (region[2] - region[0]) + region[0]
v_actual = frac(uv) * (region[3] - region[1]) + region[1]
```

Convert UV to pixel coordinates:
```python
px = (u * tex_width).astype(int)
py = ((1 - v) * tex_height).astype(int)  # V is typically flipped (0 = bottom)
```

**Output of this stage:** A set of UV-space triangles (in pixel coordinates) per node, forming a mask of the region to redact on each texture atlas.

---

## Stage 4: Redact Textures

For each affected node's texture atlas, apply the redaction.

### 4a. Load the texture

```python
from PIL import Image
texture = Image.open(f"nodes/{resource_id}/textures/0.jpg")
tex_w, tex_h = texture.size
```

Textures may be JPEG, PNG, or KTX2/Basis. For KTX2, you'll need to transcode to a raster format first (use `basis_universal` CLI or `Pillow` with appropriate plugin).

### 4b. Build a redaction mask

Rasterize all the UV-space triangles into a binary mask at the texture's resolution:

```python
import numpy as np
from PIL import ImageDraw

mask = Image.new('L', (tex_w, tex_h), 0)
draw = ImageDraw.Draw(mask)

for tri_uvs in redaction_triangles:
    # tri_uvs is [(px0,py0), (px1,py1), (px2,py2)]
    draw.polygon(tri_uvs, fill=255)

mask_np = np.array(mask)
```

**Edge handling:** Dilate the mask by a few pixels (`scipy.ndimage.binary_dilation`) to avoid sub-pixel seams where the original texture bleeds through at triangle edges.

### 4c. Apply blur or fill

**Option A — Gaussian blur (preserves rough structure, obscures detail):**
```python
from PIL import ImageFilter

blurred = texture.filter(ImageFilter.GaussianBlur(radius=20))
texture_np = np.array(texture)
blurred_np = np.array(blurred)
mask_3ch = np.stack([mask_np]*3, axis=-1) / 255.0
result = (texture_np * (1 - mask_3ch) + blurred_np * mask_3ch).astype(np.uint8)
```

**Option B — Solid fill (complete obliteration):**
```python
texture_np[mask_np > 0] = [128, 128, 128]  # neutral gray
```

**Option C — Pixelation (downscale then upscale the masked region):**
```python
# For each connected component of the mask, extract bbox, downscale 16x, upscale back
```

### 4d. Save the modified texture

```python
result_img = Image.fromarray(result)
result_img.save(f"nodes/{resource_id}/textures/0.jpg", quality=85)
```

**Critical:** Keep the exact same image dimensions and format. Do not change JPEG to PNG or vice versa — the node's metadata references the specific format and dimensions.

### 4e. Handle multiple texture formats

A single node may have the same texture in multiple formats/resolutions: `0.jpg`, `0_0_1.bin` (KTX2/Basis), `0_0_2.bin`, etc. You must redact **all** variants. For compressed formats (KTX2), you'll need to decompress → redact → recompress, or simply delete the compressed variants and fall back to JPEG (clients will use whatever is available).

---

## Stage 5: Validate

Before repacking, verify the modifications:

1. **Texture dimensions unchanged:** `Image.open(path).size == original_size` for every modified texture.
2. **Geometry buffers untouched:** Confirm you haven't accidentally modified geometry files (byte-compare against originals).
3. **Visual spot-check:** Render a few modified textures and overlay the redaction mask to confirm alignment. Matplotlib works fine for this.
4. **Node metadata intact:** Ensure `shared/` descriptors and any texture metadata JSON haven't been altered.

---

## Stage 6: Repack the SLPK

Reassemble the directory back into a valid `.slpk`.

**Action:**
1. Re-gzip any entries that were originally gzipped (geometry buffers, node pages, layer JSON). Check the original `.slpk` to determine which entries were compressed.
2. For SLPK v1.7+: regenerate the `@specialIndexFileHASH128` hash index. This is an MD5-based lookup table mapping internal paths to byte offsets. If you skip this, some clients (ArcGIS Pro, ArcGIS Online) may fail to read the SLPK. The hash computation is: `MD5(lowercase(path))` → 16-byte key, stored in a sorted binary index with 64-byte records.
3. Zip everything back using `zipfile.ZipFile` with `ZIP_STORED` (no compression) for v1.7+, or `ZIP_DEFLATED` for v1.6.
4. Ensure the archive root structure is flat (no extra parent directory).

**Hash index regeneration (v1.7+):**
```python
import hashlib, struct

entries = []
for path in all_internal_paths:
    key = hashlib.md5(path.lower().encode('utf-8')).digest()
    offset = ...  # byte offset of this entry in the zip
    size = ...    # uncompressed size
    entries.append((key, offset, size))

entries.sort(key=lambda e: e[0])  # sort by MD5 hash

# Write as binary: [count(UInt64)] + [key(16B) + offset(UInt64) + size(UInt64)] × count
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Binary buffer parsing, array ops |
| `Pillow` | Texture loading, drawing, blur |
| `scipy` | Mask dilation |
| `shapely` | Polygon intersection tests |
| `pyproj` | CRS reprojection (if redaction polygons aren't in layer CRS) |
| `rtree` | Spatial indexing of node bounding volumes |
| `DracoPy` or `draco` CLI | Decoding draco-compressed geometry (if applicable) |
| `basis_universal` CLI | KTX2/Basis texture transcoding (if applicable) |

---

## Edge Cases and Gotchas

**UV wrapping and atlas regions.** Integrated mesh textures frequently use atlas packing with `uvRegion` metadata. If `uvRegion` is present, raw UV values may exceed `[0,1]` (tiled/repeated textures). You must wrap UVs into the atlas sub-region before computing pixel coordinates. Ignoring this will cause redaction to land on the wrong part of the atlas.

**Multi-LoD consistency.** Parent nodes contain coarser versions of the same geometry. A polygon that covers a building facade at the leaf level will cover a larger, blockier region at the parent level. You must redact all LoDs, or a user zooming out will see un-redacted content.

**Seam artifacts.** Triangle edges in UV space may not align perfectly with pixel boundaries. Dilate your mask by 2–3 pixels to avoid visible seams. For blur-based redaction, use a soft (feathered) mask edge rather than a hard cutoff.

**Draco re-encoding.** If the SLPK uses draco for geometry, clients typically prefer the draco buffer. If you only need to modify textures (not geometry), you don't need to touch draco at all. But if for some reason you must modify geometry (e.g., to remove vertices), you'd need to re-encode with draco, which may alter quantization. Stick to texture-only modification.

**Very large SLPKs.** City-scale integrated meshes can be 50–200 GB. The R-tree spatial index avoids scanning all nodes, but you still need to handle streaming/chunked I/O. Process nodes in batches, don't load all textures into memory simultaneously.

**KTX2/Basis textures.** If the SLPK uses Basis Universal supercompressed textures (common in v1.8+/OGC I3S 1.2), you must transcode to RGBA, apply the redaction, then re-encode. Basis encoding is lossy and slow — consider whether you can ship JPEG fallbacks instead.

---

## Suggested Development Order

1. **Prototype on a small SLPK** — pick one with a handful of nodes, JPEG textures, and uncompressed geometry. Get the full unpack → identify → blur → repack loop working end-to-end.
2. **Add draco support** — integrate DracoPy for geometry decoding on nodes that only have compressed buffers.
3. **Add KTX2 support** — handle Basis-compressed textures.
4. **Add hash index regeneration** — needed for ArcGIS Pro/Online compatibility.
5. **Performance pass** — parallelize per-node processing, stream large SLPKs.
6. **CLI interface** — accept `--slpk`, `--polygons` (GeoJSON), `--method` (blur/fill/pixelate), `--radius` (blur kernel size).
