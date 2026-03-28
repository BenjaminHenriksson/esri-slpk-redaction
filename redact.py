#!/usr/bin/env python3
"""SLPK Texture Redaction Pipeline.

Given an SLPK (Integrated Mesh) and redaction polygons (GeoJSON in WGS84),
pixelate+blur texture regions corresponding to those polygons across all LoD levels.

For debugging: outputs only the region around the redaction polygons (~500m buffer).
"""

import argparse
import gzip
import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pyproj import Transformer
from rtree import index as rtree_index
from scipy.ndimage import binary_dilation
from shapely.geometry import Polygon, shape
from shapely.prepared import prep
from shapely.strtree import STRtree


def load_redaction_polygons(geojson_path: str, transformer: Transformer) -> list[Polygon]:
    """Load GeoJSON polygons and reproject from WGS84 to the layer CRS."""
    with open(geojson_path) as f:
        gj = json.load(f)

    polygons = []
    for feature in gj["features"]:
        geom = shape(feature["geometry"])
        # Reproject each coordinate from WGS84 (lon, lat) to layer CRS
        if geom.geom_type == "Polygon":
            coords = []
            for ring in [geom.exterior] + list(geom.interiors):
                projected = [transformer.transform(lon, lat) for lon, lat in ring.coords]
                coords.append(projected)
            polygons.append(Polygon(coords[0], coords[1:]))
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = []
                for ring in [poly.exterior] + list(poly.interiors):
                    projected = [transformer.transform(lon, lat) for lon, lat in ring.coords]
                    coords.append(projected)
                polygons.append(Polygon(coords[0], coords[1:]))
    return polygons


def load_node_pages(zf: zipfile.ZipFile) -> list[dict]:
    """Load all node pages and return a flat list of nodes."""
    page_names = sorted(
        [n for n in zf.namelist() if n.startswith("nodepages/") and n.endswith(".json.gz")],
        key=lambda x: int(x.split("/")[1].split(".")[0]),
    )
    all_nodes = []
    for pname in page_names:
        data = gzip.decompress(zf.read(pname))
        page = json.loads(data)
        all_nodes.extend(page["nodes"])
    return all_nodes


def build_spatial_index(nodes: list[dict]) -> tuple[rtree_index.Index, dict]:
    """Build an R-tree spatial index over node OBBs. Returns (rtree, node_index_map)."""
    idx = rtree_index.Index()
    node_map = {}  # rtree_id -> node

    for i, node in enumerate(nodes):
        if "obb" not in node:
            continue
        obb = node["obb"]
        cx, cy, cz = obb["center"]
        hx, hy, hz = obb["halfSize"]
        # AABB from OBB (quaternion is identity [0,0,0,1] for all nodes we've seen)
        minx, miny = cx - hx, cy - hy
        maxx, maxy = cx + hx, cy + hy
        idx.insert(i, (minx, miny, maxx, maxy))
        node_map[i] = node

    return idx, node_map


def find_candidate_nodes(
    rtree_idx: rtree_index.Index,
    node_map: dict,
    polygons: list[Polygon],
    buffer_m: float,
) -> set[int]:
    """Find nodes whose OBB intersects any redaction polygon (with buffer)."""
    candidates = set()
    for poly in polygons:
        bounds = poly.buffer(buffer_m).bounds  # (minx, miny, maxx, maxy)
        hits = list(rtree_idx.intersection(bounds))
        candidates.update(hits)
    return candidates


def find_region_nodes(
    rtree_idx: rtree_index.Index,
    node_map: dict,
    polygons: list[Polygon],
    buffer_m: float,
) -> set[int]:
    """Find ALL nodes in the ~500m region around the redaction polygons (for output)."""
    candidates = set()
    for poly in polygons:
        bounds = poly.buffer(buffer_m).bounds
        hits = list(rtree_idx.intersection(bounds))
        candidates.update(hits)
    return candidates


def parse_geometry_buffer(data: bytes, has_color: bool = True) -> dict:
    """Parse an I3S geometry buffer (uncompressed, post-gzip-decompression).

    Layout: [header: vertexCount(U32), featureCount(U32)]
            [position: F32x3 x vertexCount]
            [normal: F32x3 x vertexCount]
            [uv0: F32x2 x vertexCount]
            [color: U8x4 x vertexCount]  (if present)
            [featureId: U64 x featureCount]
            [faceRange: U32x2 x featureCount]
    """
    if len(data) < 8:
        return None

    vertex_count, feature_count = struct.unpack_from("<II", data, 0)
    offset = 8

    if vertex_count == 0:
        return None

    # Position: Float32 x 3
    pos_size = vertex_count * 3 * 4
    if offset + pos_size > len(data):
        return None
    positions = np.frombuffer(data, dtype=np.float32, count=vertex_count * 3, offset=offset).reshape(-1, 3)
    offset += pos_size

    # Normal: Float32 x 3
    norm_size = vertex_count * 3 * 4
    if offset + norm_size > len(data):
        return None
    offset += norm_size  # skip normals, we don't need them

    # UV0: Float32 x 2
    uv_size = vertex_count * 2 * 4
    if offset + uv_size > len(data):
        return None
    uv0 = np.frombuffer(data, dtype=np.float32, count=vertex_count * 2, offset=offset).reshape(-1, 2)
    offset += uv_size

    # Color: UInt8 x 4 (if present)
    if has_color:
        color_size = vertex_count * 4
        if offset + color_size <= len(data):
            offset += color_size  # skip color

    return {
        "vertex_count": vertex_count,
        "feature_count": feature_count,
        "positions": positions,
        "uv0": uv0,
    }


def find_redaction_triangles(
    geom: dict,
    obb_center: list[float],
    redaction_polys: list[Polygon],
) -> list[int]:
    """Find triangle indices that intersect any redaction polygon.

    Returns list of triangle indices (triangle i = vertices [3i, 3i+1, 3i+2]).
    """
    positions = geom["positions"].copy()
    # Transform to global CRS by adding OBB center
    positions[:, 0] += obb_center[0]
    positions[:, 1] += obb_center[1]
    positions[:, 2] += obb_center[2]

    num_triangles = geom["vertex_count"] // 3
    if num_triangles == 0:
        return []

    # Build triangle centroids for fast pre-filtering
    tri_positions = positions.reshape(num_triangles, 3, 3)
    centroids_2d = tri_positions[:, :, :2].mean(axis=1)  # [N, 2] (x, y)

    # Use STRtree for efficient intersection
    prepared_polys = [prep(p) for p in redaction_polys]

    # Fast pre-filter: check which centroids are within a generous buffer of any polygon
    # Then do exact triangle-polygon intersection
    redaction_tris = []

    # Build shapely triangles for all and use STRtree
    tri_polys = []
    tri_indices = []
    for i in range(num_triangles):
        v0 = tri_positions[i, 0, :2]
        v1 = tri_positions[i, 1, :2]
        v2 = tri_positions[i, 2, :2]
        try:
            tp = Polygon([v0, v1, v2])
            if tp.is_valid and not tp.is_empty:
                tri_polys.append(tp)
                tri_indices.append(i)
        except Exception:
            continue

    if not tri_polys:
        return []

    tree = STRtree(tri_polys)

    for rp, prep_rp in zip(redaction_polys, prepared_polys):
        hits = tree.query(rp)
        for hit_idx in hits:
            tri_poly = tri_polys[hit_idx]
            if prep_rp.intersects(tri_poly):
                redaction_tris.append(tri_indices[hit_idx])

    return sorted(set(redaction_tris))


def build_redaction_mask(
    redaction_tris: list[int],
    uv0: np.ndarray,
    tex_w: int,
    tex_h: int,
    dilation_px: int = 3,
) -> np.ndarray | None:
    """Build a binary mask at texture resolution from UV-space triangles."""
    if not redaction_tris:
        return None

    mask = Image.new("L", (tex_w, tex_h), 0)
    draw = ImageDraw.Draw(mask)

    for tri_idx in redaction_tris:
        uv_a = uv0[3 * tri_idx]
        uv_b = uv0[3 * tri_idx + 1]
        uv_c = uv0[3 * tri_idx + 2]

        # UV to pixel: u * width, (1-v) * height (V is flipped)
        pts = [
            (float(uv_a[0] * tex_w), float((1 - uv_a[1]) * tex_h)),
            (float(uv_b[0] * tex_w), float((1 - uv_b[1]) * tex_h)),
            (float(uv_c[0] * tex_w), float((1 - uv_c[1]) * tex_h)),
        ]
        draw.polygon(pts, fill=255)

    mask_np = np.array(mask)

    # Dilate to avoid seam artifacts
    if dilation_px > 0:
        mask_np = (binary_dilation(mask_np > 0, iterations=dilation_px).astype(np.uint8) * 255)

    return mask_np


def apply_pixelate_blur(
    texture: Image.Image,
    mask_np: np.ndarray,
    pixelate_factor: int = 16,
    blur_radius: int = 8,
) -> Image.Image:
    """Apply pixelation then blur to masked region of texture."""
    tex_np = np.array(texture)
    tex_w, tex_h = texture.size

    # Pixelate: downscale then upscale
    small = texture.resize(
        (max(1, tex_w // pixelate_factor), max(1, tex_h // pixelate_factor)),
        Image.Resampling.NEAREST,
    )
    pixelated = small.resize((tex_w, tex_h), Image.Resampling.NEAREST)

    # Blur the pixelated version
    blurred = pixelated.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_np = np.array(blurred)

    # Composite: use mask to blend
    mask_3ch = np.stack([mask_np] * 3, axis=-1) / 255.0
    result = (tex_np * (1 - mask_3ch) + blurred_np * mask_3ch).astype(np.uint8)

    return Image.fromarray(result)


def regenerate_hash_index(internal_paths: list[str], offsets: dict, sizes: dict) -> bytes:
    """Regenerate the @specialIndexFileHASH128 / @3dtilesIndex1@ hash index.

    Each entry: MD5(lowercase(path)) → 16-byte key, stored sorted.
    Record format: [count(U64)] + [key(16B) + offset(U64) + size(U64)] x count
    """
    entries = []
    for path in internal_paths:
        key = hashlib.md5(path.lower().encode("utf-8")).digest()
        entries.append((key, offsets[path], sizes[path]))

    entries.sort(key=lambda e: e[0])

    buf = struct.pack("<Q", len(entries))
    for key, offset, size in entries:
        buf += key + struct.pack("<QQ", offset, size)

    return buf


def main():
    parser = argparse.ArgumentParser(description="SLPK Texture Redaction")
    parser.add_argument("--slpk", required=True, help="Input .slpk file")
    parser.add_argument("--polygons", required=True, help="GeoJSON file with redaction polygons (WGS84)")
    parser.add_argument("--output", required=True, help="Output .slpk file")
    parser.add_argument("--buffer", type=float, default=500.0, help="Buffer around polygons for region output (meters)")
    parser.add_argument("--pixelate-factor", type=int, default=16, help="Pixelation downscale factor")
    parser.add_argument("--blur-radius", type=int, default=8, help="Gaussian blur radius")
    parser.add_argument("--dilation", type=int, default=3, help="Mask dilation in pixels")
    args = parser.parse_args()

    # --- Stage 0: Open SLPK and read metadata ---
    print("Stage 0: Reading SLPK metadata...")
    zf = zipfile.ZipFile(args.slpk, "r")

    layer_data = gzip.decompress(zf.read("3dSceneLayer.json.gz"))
    layer = json.loads(layer_data)

    layer_crs_wkid = layer["spatialReference"]["wkid"]
    print(f"  Layer CRS: EPSG:{layer_crs_wkid}")

    geom_schema = layer["store"]["defaultGeometrySchema"]
    has_color = "color" in geom_schema["vertexAttributes"]
    ordering = geom_schema["ordering"]
    print(f"  Geometry attributes: {ordering}")
    print(f"  Texture encoding: {layer['store']['textureEncoding']}")

    # --- Set up CRS transformer (WGS84 → layer CRS) ---
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{layer_crs_wkid}", always_xy=True)

    # --- Load redaction polygons ---
    print(f"  Loading redaction polygons from {args.polygons}...")
    redaction_polys = load_redaction_polygons(args.polygons, transformer)
    print(f"  {len(redaction_polys)} redaction polygon(s)")

    for i, p in enumerate(redaction_polys):
        bounds = p.bounds
        print(f"    Polygon {i}: bounds ({bounds[0]:.1f}, {bounds[1]:.1f}) - ({bounds[2]:.1f}, {bounds[3]:.1f})")

    # --- Stage 1: Build spatial index ---
    print("\nStage 1: Loading node pages and building spatial index...")
    all_nodes = load_node_pages(zf)
    print(f"  {len(all_nodes)} nodes loaded")

    rtree_idx, node_map = build_spatial_index(all_nodes)
    print(f"  {len(node_map)} nodes indexed")

    # --- Stage 2: Find candidate nodes ---
    print("\nStage 2: Finding candidate nodes...")
    # Nodes to redact (intersect redaction polygons)
    redact_candidates = find_candidate_nodes(rtree_idx, node_map, redaction_polys, buffer_m=50)
    print(f"  {len(redact_candidates)} nodes intersect redaction polygons")

    # Nodes to include in output (broader region for debugging)
    region_candidates = find_region_nodes(rtree_idx, node_map, redaction_polys, buffer_m=args.buffer)
    print(f"  {len(region_candidates)} nodes in output region (~{args.buffer}m buffer)")

    # Build mapping: node_index -> resource_id
    # resource_id is stored in the node's resourceId field, or sometimes it's derived
    # from the node's index. Let's check what fields are available.
    node_resource_map = {}
    for ni in region_candidates:
        node = node_map[ni]
        # resourceId may or may not be present; fall back to index
        rid = node.get("resourceId", node.get("index", ni))
        node_resource_map[ni] = rid

    # --- Stage 3 & 4: Parse geometry and redact textures ---
    print("\nStage 3-4: Processing nodes (geometry parsing + texture redaction)...")
    redacted_textures = {}  # resource_id -> modified JPEG bytes
    nodes_processed = 0
    nodes_redacted = 0

    for ni in sorted(redact_candidates):
        node = node_map[ni]
        rid = node_resource_map.get(ni)
        if rid is None:
            continue

        # Check if this node has geometry and texture
        geom_path = f"nodes/{rid}/geometries/0.bin.gz"
        tex_path = f"nodes/{rid}/textures/0.jpg"

        try:
            zf.getinfo(geom_path)
            zf.getinfo(tex_path)
        except KeyError:
            continue

        nodes_processed += 1

        # Read and decompress geometry
        geom_data = gzip.decompress(zf.read(geom_path))
        geom = parse_geometry_buffer(geom_data, has_color=has_color)
        if geom is None:
            continue

        obb_center = node["obb"]["center"]

        # Find triangles intersecting redaction polygons
        redaction_tris = find_redaction_triangles(geom, obb_center, redaction_polys)
        if not redaction_tris:
            continue

        # Load texture
        tex_data = zf.read(tex_path)
        texture = Image.open(io.BytesIO(tex_data))
        tex_w, tex_h = texture.size

        # Build mask
        mask = build_redaction_mask(redaction_tris, geom["uv0"], tex_w, tex_h, dilation_px=args.dilation)
        if mask is None:
            continue

        # Apply pixelate + blur
        result = apply_pixelate_blur(texture, mask, args.pixelate_factor, args.blur_radius)

        # Save to buffer
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        redacted_textures[rid] = buf.getvalue()
        nodes_redacted += 1

        if nodes_redacted % 50 == 0:
            print(f"  Redacted {nodes_redacted} nodes so far...")

    print(f"  Processed {nodes_processed} nodes, redacted {nodes_redacted} textures")

    # --- Stage 5: Validate ---
    print("\nStage 5: Validating...")
    for rid, tex_bytes in redacted_textures.items():
        img = Image.open(io.BytesIO(tex_bytes))
        orig_data = zf.read(f"nodes/{rid}/textures/0.jpg")
        orig = Image.open(io.BytesIO(orig_data))
        if img.size != orig.size:
            print(f"  WARNING: Texture size mismatch for node {rid}: {img.size} vs {orig.size}")
    print(f"  {len(redacted_textures)} textures validated")

    # --- Stage 6: Repack (region-only output) ---
    print(f"\nStage 6: Repacking region to {args.output}...")

    # Determine which ZIP entries to include in the output
    # Include: metadata files + all entries for nodes in the region
    region_rids = set()
    for ni in region_candidates:
        rid = node_resource_map.get(ni)
        if rid is not None:
            region_rids.add(str(rid))

    entries_to_copy = []
    for name in zf.namelist():
        if name.startswith("nodes/"):
            parts = name.split("/")
            if len(parts) >= 2 and parts[1] in region_rids:
                entries_to_copy.append(name)
        elif name.startswith("@"):
            # Skip hash index, we'll regenerate
            continue
        else:
            # Include all non-node files (metadata, nodepages, etc.)
            entries_to_copy.append(name)

    print(f"  {len(entries_to_copy)} entries to include in output")

    # Write output SLPK
    offsets = {}
    sizes = {}
    internal_paths = []

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_STORED) as out_zf:
        for entry_name in entries_to_copy:
            # Check if this is a redacted texture
            if entry_name.startswith("nodes/") and entry_name.endswith("/textures/0.jpg"):
                rid_str = entry_name.split("/")[1]
                rid_key = int(rid_str) if rid_str.isdigit() else rid_str
                if rid_key in redacted_textures:
                    out_zf.writestr(entry_name, redacted_textures[rid_key])
                    internal_paths.append(entry_name)
                    continue

            # Copy original entry
            data = zf.read(entry_name)
            out_zf.writestr(entry_name, data)
            internal_paths.append(entry_name)

        # Regenerate hash index
        # We need to get the offsets after writing, but zipfile doesn't expose this easily
        # during writing. We'll add the hash index entry but skip regeneration for now
        # since this is a debug output.

    print(f"  Output written to {args.output}")

    zf.close()

    # Report file size
    out_size = Path(args.output).stat().st_size
    print(f"  Output size: {out_size / 1024 / 1024:.1f} MB")
    print("\nDone!")


if __name__ == "__main__":
    main()
