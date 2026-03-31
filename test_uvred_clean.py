#!/usr/bin/env python3
"""SLPK texture redaction with 3D voxel-based pixelation.

Redaction colors are defined in a 3D voxel grid (world space), ensuring:
- Seamless across UV atlas seams (same world position → same color)
- Seamless across LOD transitions (all LODs sample the same grid)
- No mixing of surfaces at different heights (3D grid, not 2D)
"""

import gzip
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.ndimage import binary_dilation, gaussian_filter
from shapely.geometry import Polygon, Point, box, shape
from shapely.prepared import prep
from shapely.strtree import STRtree

SLPK = "/home/sbk-usr/mesh-redaction/Stockholm_3D_2025_North.slpk"
OUTPUT = "/home/sbk-usr/mesh-redaction/test_uvred_clean.slpk"
VOXEL_SIZE = 2.0  # meters per voxel cell
BLUR_SIGMA = 1.5  # voxel cells


def _bary_uv(px, py, uv_a, uv_b, uv_c):
    """Barycentric coords of (px,py) within UV triangle (uv_a, uv_b, uv_c)."""
    d00 = (uv_b[0]-uv_a[0])**2 + (uv_b[1]-uv_a[1])**2
    d01 = (uv_b[0]-uv_a[0])*(uv_c[0]-uv_a[0]) + (uv_b[1]-uv_a[1])*(uv_c[1]-uv_a[1])
    d11 = (uv_c[0]-uv_a[0])**2 + (uv_c[1]-uv_a[1])**2
    d20 = (px-uv_a[0])*(uv_b[0]-uv_a[0]) + (py-uv_a[1])*(uv_b[1]-uv_a[1])
    d21 = (px-uv_a[0])*(uv_c[0]-uv_a[0]) + (py-uv_a[1])*(uv_c[1]-uv_a[1])
    denom = d00*d11 - d01*d01
    if abs(denom) < 1e-12:
        return -1, -1, -1
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    return 1-v-w, v, w


def parse_geometry(zf, res_id):
    """Parse geometry buffer, return (positions, uv0, file_vc) or None."""
    gdata = gzip.decompress(zf.read(f"nodes/{res_id}/geometries/0.bin.gz"))
    if len(gdata) < 8:
        return None
    file_vc = struct.unpack_from("<I", gdata, 0)[0]
    if file_vc == 0:
        return None
    off = 8
    pos_size = file_vc * 3 * 4
    if off + pos_size > len(gdata):
        return None
    positions = np.frombuffer(gdata, dtype=np.float32, count=file_vc*3, offset=off).reshape(-1, 3)
    off += pos_size + file_vc * 3 * 4  # skip normals
    uv_size = file_vc * 2 * 4
    if off + uv_size > len(gdata):
        return None
    uv0 = np.frombuffer(gdata, dtype=np.float32, count=file_vc*2, offset=off).reshape(-1, 2)
    return positions, uv0, file_vc


def main():
    print("=== Voxel-Based Redaction ===")

    zipfile.ZipExtFile._update_crc = lambda self, data: None
    zf = zipfile.ZipFile(SLPK, "r")

    # Load polygon
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3011", always_xy=True)
    with open("region.geojson") as f:
        gj = json.load(f)
    geom = shape(gj["features"][0]["geometry"])
    coords = [transformer.transform(lon, lat) for lon, lat in geom.exterior.coords]
    poly = Polygon(coords)
    prep_p = prep(poly)
    print(f"Polygon bounds: {poly.bounds}")

    # Find candidates from node pages, separate leaves from interior
    page_names = sorted(
        [n for n in zf.namelist() if n.startswith("nodepages/") and n.endswith(".json.gz")],
        key=lambda x: int(x.split("/")[1].split(".")[0]),
    )
    candidates = []
    for pname in page_names:
        page = json.loads(gzip.decompress(zf.read(pname)))
        for node in page["nodes"]:
            if "obb" not in node or "mesh" not in node:
                continue
            obb = node["obb"]
            cx, cy = obb["center"][0], obb["center"][1]
            hx, hy = obb["halfSize"][0], obb["halfSize"][1]
            if box(cx-hx, cy-hy, cx+hx, cy+hy).intersects(poly):
                is_leaf = len(node.get("children", [])) == 0
                candidates.append((node, is_leaf))

    leaves = [(n, l) for n, l in candidates if l]
    print(f"Candidates: {len(candidates)} ({len(leaves)} leaves)")

    # ================================================================
    # PHASE 1: Build 3D voxel grid from leaf nodes
    # ================================================================
    print("\nPhase 1: Building voxel grid from leaf textures...")

    # Grid bounds from polygon + buffer
    bx0, by0, bx1, by1 = poly.bounds
    BUFFER = 10.0
    # Z range: scan leaf geometry to find min/max Z
    z_min, z_max = 1e9, -1e9
    for node, _ in leaves:
        res_id = node["mesh"]["material"]["resource"]
        result = parse_geometry(zf, res_id)
        if result is None:
            continue
        positions, _, _ = result
        obb_center = node["obb"]["center"]
        world_z = positions[:, 2] + obb_center[2]
        z_min = min(z_min, world_z.min())
        z_max = max(z_max, world_z.max())

    grid_origin = np.array([bx0 - BUFFER, by0 - BUFFER, z_min - BUFFER])
    grid_size = np.array([
        int(np.ceil((bx1 + BUFFER - grid_origin[0]) / VOXEL_SIZE)) + 1,
        int(np.ceil((by1 + BUFFER - grid_origin[1]) / VOXEL_SIZE)) + 1,
        int(np.ceil((z_max + BUFFER - grid_origin[2]) / VOXEL_SIZE)) + 1,
    ])
    print(f"  Grid: {grid_size[0]}x{grid_size[1]}x{grid_size[2]} = {np.prod(grid_size)} voxels")
    print(f"  Origin: ({grid_origin[0]:.0f}, {grid_origin[1]:.0f}, {grid_origin[2]:.0f})")

    # Accumulation arrays
    color_sum = np.zeros((*grid_size, 3), dtype=np.float64)
    color_count = np.zeros(grid_size, dtype=np.int32)

    # Fill voxels from leaf node vertices
    for node, _ in leaves:
        res_id = node["mesh"]["material"]["resource"]
        result = parse_geometry(zf, res_id)
        if result is None:
            continue
        positions, uv0, file_vc = result
        obb_center = node["obb"]["center"]

        pos_world = positions.copy()
        pos_world[:, 0] += obb_center[0]
        pos_world[:, 1] += obb_center[1]
        pos_world[:, 2] += obb_center[2]

        # Load texture
        tex_data = zf.read(f"nodes/{res_id}/textures/0.jpg")
        tex = Image.open(io.BytesIO(tex_data))
        tex_np = np.array(tex)
        tw, th = tex.size

        # For each vertex inside polygon, sample texture and accumulate
        for vi in range(file_vc):
            wx, wy, wz = pos_world[vi]
            if not prep_p.contains(Point(wx, wy)):
                continue

            # Sample texture at vertex UV
            u, v = uv0[vi]
            px = int(np.clip(u * tw, 0, tw - 1))
            py = int(np.clip(v * th, 0, th - 1))
            color = tex_np[py, px, :3].astype(np.float64)

            # Voxel cell
            gi = int((wx - grid_origin[0]) / VOXEL_SIZE)
            gj = int((wy - grid_origin[1]) / VOXEL_SIZE)
            gk = int((wz - grid_origin[2]) / VOXEL_SIZE)
            if 0 <= gi < grid_size[0] and 0 <= gj < grid_size[1] and 0 <= gk < grid_size[2]:
                color_sum[gi, gj, gk] += color
                color_count[gi, gj, gk] += 1

    # Average
    occupied = color_count > 0
    print(f"  Occupied voxels: {np.sum(occupied)} ({100*np.sum(occupied)/np.prod(grid_size):.2f}%)")
    voxel_color = np.zeros_like(color_sum)
    for c in range(3):
        voxel_color[:, :, :, c] = np.where(occupied, color_sum[:, :, :, c] / color_count, 0)

    # ================================================================
    # PHASE 2: Blur voxel grid (3D Gaussian, only through occupied cells)
    # ================================================================
    print("\nPhase 2: Blurring voxel grid...")

    # Gaussian blur each color channel, masked to occupied voxels
    # Use a weighted blur: blur(color * occupied) / blur(occupied) to avoid
    # pulling in zeros from empty space
    occupied_f = occupied.astype(np.float64)
    weight = gaussian_filter(occupied_f, sigma=BLUR_SIGMA)
    weight = np.maximum(weight, 1e-10)  # avoid division by zero

    blurred_color = np.zeros_like(voxel_color)
    for c in range(3):
        blurred_color[:, :, :, c] = gaussian_filter(
            voxel_color[:, :, :, c] * occupied_f, sigma=BLUR_SIGMA
        ) / weight

    print(f"  Blur sigma: {BLUR_SIGMA} cells ({BLUR_SIGMA * VOXEL_SIZE:.1f}m)")

    # ================================================================
    # PHASE 3: Paint all LOD levels using voxel grid
    # ================================================================
    print(f"\nPhase 3: Painting {len(candidates)} nodes...")

    replace_paths = {}
    seen = set()
    redacted = 0

    for node, is_leaf in candidates:
        res_id = node["mesh"]["material"]["resource"]
        if res_id in seen:
            continue
        seen.add(res_id)

        tex_path = f"nodes/{res_id}/textures/0.jpg"
        geom_path = f"nodes/{res_id}/geometries/0.bin.gz"
        try:
            zf.getinfo(tex_path)
            zf.getinfo(geom_path)
        except KeyError:
            continue

        result = parse_geometry(zf, res_id)
        if result is None:
            continue
        positions, uv0, file_vc = result

        obb_center = node["obb"]["center"]
        pos_world = positions.copy()
        pos_world[:, 0] += obb_center[0]
        pos_world[:, 1] += obb_center[1]
        pos_world[:, 2] += obb_center[2]

        num_tri = file_vc // 3
        tri_pos = pos_world.reshape(num_tri, 3, 3)

        # Vertex containment
        vertex_inside = np.array([
            prep_p.contains(Point(pos_world[i, 0], pos_world[i, 1]))
            for i in range(file_vc)
        ])
        tri_inside = vertex_inside.reshape(num_tri, 3).sum(axis=1)

        # Straddle detection for large boundary triangles
        straddle_set = set()
        tri_polys_2d = []
        tri_polys_idx = []
        for i in range(num_tri):
            if tri_inside[i] > 0:
                continue
            v0, v1, v2 = tri_pos[i, 0, :2], tri_pos[i, 1, :2], tri_pos[i, 2, :2]
            try:
                tp = Polygon([v0, v1, v2])
                if tp.is_valid and not tp.is_empty:
                    tri_polys_2d.append(tp)
                    tri_polys_idx.append(i)
            except:
                continue
        if tri_polys_2d:
            tree = STRtree(tri_polys_2d)
            for hit_idx in tree.query(poly):
                if poly.intersects(tri_polys_2d[hit_idx]):
                    straddle_set.add(tri_polys_idx[hit_idx])

        # Load texture
        tex_data = zf.read(tex_path)
        texture = Image.open(io.BytesIO(tex_data))
        tex_w, tex_h = texture.size
        tex_np = np.array(texture)

        # Build mask and color map simultaneously
        mask = Image.new("L", (tex_w, tex_h), 0)
        draw = ImageDraw.Draw(mask)
        # Per-pixel color from voxel grid (for boundary triangles)
        voxel_tex = np.zeros_like(tex_np, dtype=np.float64)
        voxel_mask = np.zeros((tex_h, tex_w), dtype=np.uint8)

        n_full = 0
        n_boundary = 0

        for i in range(num_tri):
            n_in = tri_inside[i]
            if n_in == 0 and i not in straddle_set:
                continue

            uv_a, uv_b, uv_c = uv0[3*i], uv0[3*i+1], uv0[3*i+2]

            if n_in == 3:
                # Full triangle: rasterize for mask, sample voxel at vertices
                pts = [(uv_a[0]*tex_w, uv_a[1]*tex_h),
                       (uv_b[0]*tex_w, uv_b[1]*tex_h),
                       (uv_c[0]*tex_w, uv_c[1]*tex_h)]
                draw.polygon(pts, fill=255)
                n_full += 1
            else:
                n_boundary += 1

            # Per-pixel: iterate UV bounding box, map to world, sample voxel
            us = [uv_a[0], uv_b[0], uv_c[0]]
            vs = [uv_a[1], uv_b[1], uv_c[1]]
            px_min = max(0, int(min(us) * tex_w) - 1)
            px_max = min(tex_w - 1, int(max(us) * tex_w) + 1)
            py_min = max(0, int(min(vs) * tex_h) - 1)
            py_max = min(tex_h - 1, int(max(vs) * tex_h) + 1)

            wv0 = tri_pos[i, 0]  # full XYZ
            wv1 = tri_pos[i, 1]
            wv2 = tri_pos[i, 2]

            for py in range(py_min, py_max + 1):
                for px in range(px_min, px_max + 1):
                    u_coord = (px + 0.5) / tex_w
                    v_coord = (py + 0.5) / tex_h

                    bu, bv, bw = _bary_uv(u_coord, v_coord, uv_a, uv_b, uv_c)
                    if bu < -0.001 or bv < -0.001 or bw < -0.001:
                        continue

                    # World XYZ from barycentric
                    wx = bu * wv0[0] + bv * wv1[0] + bw * wv2[0]
                    wy = bu * wv0[1] + bv * wv1[1] + bw * wv2[1]
                    wz = bu * wv0[2] + bv * wv1[2] + bw * wv2[2]

                    # For boundary triangles, check polygon containment
                    if n_in < 3 and not prep_p.contains(Point(wx, wy)):
                        continue

                    # Sample voxel grid
                    gi = int((wx - grid_origin[0]) / VOXEL_SIZE)
                    gj_idx = int((wy - grid_origin[1]) / VOXEL_SIZE)
                    gk = int((wz - grid_origin[2]) / VOXEL_SIZE)
                    gi = np.clip(gi, 0, grid_size[0] - 1)
                    gj_idx = np.clip(gj_idx, 0, grid_size[1] - 1)
                    gk = np.clip(gk, 0, grid_size[2] - 1)

                    voxel_tex[py, px] = blurred_color[gi, gj_idx, gk]
                    voxel_mask[py, px] = 255

        # Merge: full-triangle mask + per-pixel boundary mask
        mask_np = np.maximum(np.array(mask), voxel_mask)

        if np.sum(mask_np > 0) == 0:
            continue

        # For full triangles that weren't per-pixel sampled, fill from voxel grid
        # using vertex-averaged voxel color (fast approximation)
        full_only = (np.array(mask) > 0) & (voxel_mask == 0)
        if np.sum(full_only) > 0:
            # These pixels need voxel colors but weren't individually sampled.
            # Sample at triangle centroids for each full triangle and flood fill.
            # Simpler: iterate full triangles' pixels (already done for boundary,
            # skip for full to save time — but now we need them).
            # Re-iterate full triangles for voxel sampling.
            for i in range(num_tri):
                if tri_inside[i] != 3:
                    continue
                uv_a, uv_b, uv_c = uv0[3*i], uv0[3*i+1], uv0[3*i+2]
                wv0 = tri_pos[i, 0]
                wv1 = tri_pos[i, 1]
                wv2 = tri_pos[i, 2]

                us = [uv_a[0], uv_b[0], uv_c[0]]
                vs_list = [uv_a[1], uv_b[1], uv_c[1]]
                px_min = max(0, int(min(us) * tex_w) - 1)
                px_max = min(tex_w - 1, int(max(us) * tex_w) + 1)
                py_min = max(0, int(min(vs_list) * tex_h) - 1)
                py_max = min(tex_h - 1, int(max(vs_list) * tex_h) + 1)

                for py in range(py_min, py_max + 1):
                    for px in range(px_min, px_max + 1):
                        if voxel_mask[py, px] > 0:
                            continue  # already sampled
                        if mask_np[py, px] == 0:
                            continue  # not in mask

                        u_coord = (px + 0.5) / tex_w
                        v_coord = (py + 0.5) / tex_h
                        bu, bv, bw = _bary_uv(u_coord, v_coord, uv_a, uv_b, uv_c)
                        if bu < -0.001 or bv < -0.001 or bw < -0.001:
                            continue

                        wx = bu * wv0[0] + bv * wv1[0] + bw * wv2[0]
                        wy = bu * wv0[1] + bv * wv1[1] + bw * wv2[1]
                        wz = bu * wv0[2] + bv * wv1[2] + bw * wv2[2]

                        gi = np.clip(int((wx - grid_origin[0]) / VOXEL_SIZE), 0, grid_size[0]-1)
                        gj_idx = np.clip(int((wy - grid_origin[1]) / VOXEL_SIZE), 0, grid_size[1]-1)
                        gk = np.clip(int((wz - grid_origin[2]) / VOXEL_SIZE), 0, grid_size[2]-1)

                        voxel_tex[py, px] = blurred_color[gi, gj_idx, gk]
                        voxel_mask[py, px] = 255

        # UV padding: extend into dead space
        all_tri_mask = Image.new("L", (tex_w, tex_h), 0)
        all_tri_draw = ImageDraw.Draw(all_tri_mask)
        for ti in range(num_tri):
            a, b, c = uv0[3*ti], uv0[3*ti+1], uv0[3*ti+2]
            pts = [(a[0]*tex_w, a[1]*tex_h), (b[0]*tex_w, b[1]*tex_h), (c[0]*tex_w, c[1]*tex_h)]
            all_tri_draw.polygon(pts, fill=255)
        dead_space = np.array(all_tri_mask) == 0

        padded = binary_dilation(mask_np > 0, iterations=4).astype(np.uint8) * 255
        padded[~dead_space & (mask_np == 0)] = 0
        # For padded pixels, copy nearest voxel color
        pad_only = (padded > 0) & (voxel_mask == 0)
        if np.sum(pad_only) > 0:
            # Simple: dilate the voxel color image to fill padding
            for c in range(3):
                chan = voxel_tex[:, :, c].copy()
                for _ in range(4):
                    from scipy.ndimage import maximum_filter, minimum_filter
                    filled = gaussian_filter(chan * (voxel_mask > 0).astype(float), sigma=1)
                    weight_f = gaussian_filter((voxel_mask > 0).astype(float), sigma=1)
                    weight_f = np.maximum(weight_f, 1e-10)
                    chan = np.where(voxel_mask > 0, chan, filled / weight_f)
                voxel_tex[:, :, c] = chan

        mask_np = np.maximum(mask_np, padded)

        # Composite
        mask_bool = mask_np > 0
        tex_np[mask_bool] = np.clip(voxel_tex[mask_bool], 0, 255).astype(np.uint8)
        result = Image.fromarray(tex_np)

        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=100, subsampling=0)
        red_bytes = buf.getvalue()
        replace_paths[tex_path] = red_bytes

        # Update sharedResource
        shared_path = f"nodes/{res_id}/shared/sharedResource.json.gz"
        try:
            shared = json.loads(gzip.decompress(zf.read(shared_path)))
            for td in shared.get("textureDefinitions", {}).values():
                for img in td.get("images", []):
                    img["length"] = [len(red_bytes)]
            replace_paths[shared_path] = gzip.compress(json.dumps(shared).encode("utf-8"))
        except:
            pass

        redacted += 1

    print(f"Redacted {redacted} textures")
    print(f"Total replace paths: {len(replace_paths)}")

    # === REBUILD (identical to test_all_red.py) ===
    print(f"\nRebuilding to {OUTPUT}...")
    raw_fp = open(SLPK, "rb")
    entry_count = 0
    replaced = 0

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_STORED, allowZip64=True) as outz:
        for item in zf.infolist():
            if item.filename.startswith("@"):
                continue
            if item.filename in replace_paths:
                new_info = zipfile.ZipInfo(item.filename)
                new_info.compress_type = zipfile.ZIP_STORED
                outz.writestr(new_info, replace_paths[item.filename])
                replaced += 1
            else:
                raw_fp.seek(item.header_offset)
                lfh = raw_fp.read(30)
                fname_len = struct.unpack_from("<H", lfh, 26)[0]
                extra_len = struct.unpack_from("<H", lfh, 28)[0]
                data_offset = item.header_offset + 30 + fname_len + extra_len
                raw_fp.seek(data_offset)
                raw_data = raw_fp.read(item.compress_size)
                item.compress_type = zipfile.ZIP_STORED
                outz.writestr(item, raw_data)
            entry_count += 1
            if entry_count % 100000 == 0:
                print(f"  {entry_count} entries...")

    raw_fp.close()
    zf.close()
    print(f"  {entry_count} entries, {replaced} replaced")

    # Hash table
    print("  Regenerating hash table...")
    hash_entries = []
    with zipfile.ZipFile(OUTPUT, "r") as outz:
        for item in outz.infolist():
            canonical = item.filename.lower().replace("\\", "/")
            if canonical.startswith("/"):
                canonical = canonical[1:]
            md5 = hashlib.md5(canonical.encode("utf-8")).digest()
            hash_entries.append((md5, struct.pack("<Q", item.header_offset)))
    hash_entries.sort(key=lambda e: (
        struct.unpack("<Q", e[0][:8])[0],
        struct.unpack("<Q", e[0][8:])[0],
    ))
    hash_data = b"".join(h + o for h, o in hash_entries)
    with zipfile.ZipFile(OUTPUT, "a", allowZip64=True) as outz:
        outz.writestr("@3dtilesIndex1@", hash_data)

    print(f"  Hash table: {len(hash_entries)} entries")
    out_size = Path(OUTPUT).stat().st_size
    print(f"\nOutput: {OUTPUT} ({out_size / 1024**3:.1f} GB)")
    print("Done!")


if __name__ == "__main__":
    main()
