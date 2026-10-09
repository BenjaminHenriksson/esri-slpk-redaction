#!/usr/bin/env python3
"""SLPK texture redaction with 3D voxel-based pixelation.

Redaction colors are defined in a 3D voxel grid (world space), ensuring:
- Seamless across UV atlas seams (same world position → same color)
- Seamless across LOD transitions (all LODs sample the same grid)
- No mixing of surfaces at different heights (3D grid, not 2D)
"""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from shapely.geometry import Polygon, Point, box, shape
from shapely.ops import transform as shapely_transform, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

DEFAULT_VOXEL_SIZE = 1.5  # meters per voxel cell
DEFAULT_BLUR_SIGMA = 0.5  # voxel cells (1.5 x 0.5 = 0.75 m physical blur radius)
BLACK_THRESHOLD = 3  # max channel value treated as an unfilled (black) painted pixel


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


def raw_entry_read(zf, raw_fp, name):
    """Read a stored (uncompressed) zip entry directly via its raw file offset.

    zipfile.ZipFile.read() can raise OSError [Errno 22] on Windows for entries
    located past the 4GB mark in archives larger than 4GB. SLPK entries are
    always stored uncompressed, so reading the raw bytes directly (same trick
    used when rebuilding the output archive below) sidesteps that entirely.
    """
    info = zf.getinfo(name)
    raw_fp.seek(info.header_offset)
    lfh = raw_fp.read(30)
    fname_len = struct.unpack_from("<H", lfh, 26)[0]
    extra_len = struct.unpack_from("<H", lfh, 28)[0]
    data_offset = info.header_offset + 30 + fname_len + extra_len
    raw_fp.seek(data_offset)
    return raw_fp.read(info.compress_size)


def parse_geometry(zf, raw_fp, res_id):
    """Parse geometry buffer, return (positions, uv0, file_vc) or None."""
    gdata = gzip.decompress(raw_entry_read(zf, raw_fp, f"nodes/{res_id}/geometries/0.bin.gz"))
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
    parser = argparse.ArgumentParser(description="SLPK texture redaction via 3D voxel blur")
    parser.add_argument("--slpk", required=True, help="Input SLPK file")
    parser.add_argument("--polygons", required=True, help="GeoJSON redaction polygons")
    parser.add_argument("--output", required=True, help="Output SLPK file")
    parser.add_argument("--crs", default="EPSG:3011", help="CRS of SLPK vertex coordinates (default: EPSG:3011)")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, help=f"Voxel size in meters (default: {DEFAULT_VOXEL_SIZE})")
    parser.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA, help=f"Blur sigma in voxel cells (default: {DEFAULT_BLUR_SIGMA})")
    args = parser.parse_args()

    voxel_size = args.voxel_size
    blur_sigma = args.blur_sigma

    print("=== Voxel-Based Redaction ===")

    zipfile.ZipExtFile._update_crc = lambda self, data: None
    zf = zipfile.ZipFile(args.slpk, "r")
    raw_fp = open(args.slpk, "rb")

    # Load and project all redaction polygons
    transformer = Transformer.from_crs("EPSG:4326", args.crs, always_xy=True)
    with open(args.polygons) as f:
        gj = json.load(f)

    # Accept FeatureCollection, single Feature, or bare Geometry
    if gj.get("type") == "FeatureCollection":
        features = gj.get("features", [])
    elif gj.get("type") == "Feature":
        features = [gj]
    else:
        features = [{"geometry": gj}]

    projected = []
    for feat in features:
        geom = shape(feat["geometry"])
        if geom.is_empty:
            continue
        feat_id = feat.get("properties", {}).get("ID")
        projected.append((feat_id, shapely_transform(transformer.transform, geom)))

    if not projected:
        print("No valid geometries in GeoJSON — copying input unchanged.")
        zf.close()
        raw_fp.close()
        shutil.copy2(args.slpk, args.output)
        return

    # Decompose into individual simple polygons (flatten MultiPolygons)
    individual_polys = []
    individual_ids = []
    for feat_id, geom in projected:
        if geom.geom_type == 'Polygon':
            individual_polys.append(geom)
            individual_ids.append(feat_id)
        elif geom.geom_type == 'MultiPolygon':
            individual_polys.extend(geom.geoms)
            individual_ids.extend([feat_id] * len(geom.geoms))

    if not individual_polys:
        print("No polygon geometries found — copying input unchanged.")
        zf.close()
        raw_fp.close()
        shutil.copy2(args.slpk, args.output)
        return

    # Union for fast candidate detection across all polygons
    poly = unary_union(individual_polys)
    print(f"Loaded {len(individual_polys)} polygon(s), union bounds: {poly.bounds}")

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

    if not candidates:
        print("No mesh nodes intersect the redaction polygon(s) — copying input unchanged.")
        zf.close()
        raw_fp.close()
        shutil.copy2(args.slpk, args.output)
        return

    # ================================================================
    # Process each polygon independently (own voxel grid per polygon)
    # ================================================================
    replace_paths = {}
    total_redacted = 0
    BUFFER = 10.0

    for pi, single_poly in enumerate(individual_polys):
        prep_single = prep(single_poly)
        poly_id = individual_ids[pi]
        print(f"\n--- Polygon {pi+1}/{len(individual_polys)} (ID={poly_id}) ---")

        # Find candidates for this specific polygon
        poly_candidates = []
        for node, is_leaf in candidates:
            obb = node["obb"]
            cx, cy = obb["center"][0], obb["center"][1]
            hx, hy = obb["halfSize"][0], obb["halfSize"][1]
            if box(cx-hx, cy-hy, cx+hx, cy+hy).intersects(single_poly):
                poly_candidates.append((node, is_leaf))
        poly_leaves = [(n, l) for n, l in poly_candidates if l]

        if not poly_candidates:
            print("  No intersecting nodes, skipping.")
            continue
        print(f"  Candidates: {len(poly_candidates)} ({len(poly_leaves)} leaves)")

        # === PHASE 1: Build voxel grid from leaf nodes ===
        print("  Phase 1: Building voxel grid...")
        bx0, by0, bx1, by1 = single_poly.bounds
        z_min, z_max = 1e9, -1e9
        for node, _ in poly_leaves:
            res_id = node["mesh"]["material"]["resource"]
            result = parse_geometry(zf, raw_fp, res_id)
            if result is None:
                continue
            positions, _, _ = result
            obb_center = node["obb"]["center"]
            world_z = positions[:, 2] + obb_center[2]
            z_min = min(z_min, world_z.min())
            z_max = max(z_max, world_z.max())

        if z_min > z_max:
            print("    No valid leaf geometry, skipping.")
            continue

        grid_origin = np.array([bx0 - BUFFER, by0 - BUFFER, z_min - BUFFER])
        grid_size = np.array([
            int(np.ceil((bx1 + BUFFER - grid_origin[0]) / voxel_size)) + 1,
            int(np.ceil((by1 + BUFFER - grid_origin[1]) / voxel_size)) + 1,
            int(np.ceil((z_max + BUFFER - grid_origin[2]) / voxel_size)) + 1,
        ])
        print(f"    Grid: {grid_size[0]}x{grid_size[1]}x{grid_size[2]} = {np.prod(grid_size)} voxels")

        color_sum = np.zeros((*grid_size, 3), dtype=np.float64)
        color_count = np.zeros(grid_size, dtype=np.int32)

        num_poly_leaves = len(poly_leaves)
        for leaf_idx, (node, _) in enumerate(poly_leaves):
            res_id = node["mesh"]["material"]["resource"]
            result = parse_geometry(zf, raw_fp, res_id)
            if result is None:
                continue
            positions, uv0, file_vc = result
            obb_center = node["obb"]["center"]

            pos_world = positions.copy()
            pos_world[:, 0] += obb_center[0]
            pos_world[:, 1] += obb_center[1]
            pos_world[:, 2] += obb_center[2]

            print(f"    [poly ID={poly_id}] leaf {leaf_idx}/{num_poly_leaves} res_id={res_id} reading texture...", flush=True)
            tex_data = raw_entry_read(zf, raw_fp, f"nodes/{res_id}/textures/0.jpg")
            tex = Image.open(io.BytesIO(tex_data))
            tex_np = np.array(tex)
            tw, th = tex.size

            for vi in range(file_vc):
                wx, wy, wz = pos_world[vi]
                if not prep_single.contains(Point(wx, wy)):
                    continue
                u, v = uv0[vi]
                px = int(np.clip(u * tw, 0, tw - 1))
                py = int(np.clip(v * th, 0, th - 1))
                color = tex_np[py, px, :3].astype(np.float64)
                gi = int((wx - grid_origin[0]) / voxel_size)
                gj = int((wy - grid_origin[1]) / voxel_size)
                gk = int((wz - grid_origin[2]) / voxel_size)
                if 0 <= gi < grid_size[0] and 0 <= gj < grid_size[1] and 0 <= gk < grid_size[2]:
                    color_sum[gi, gj, gk] += color
                    color_count[gi, gj, gk] += 1

        occupied = color_count > 0
        print(f"    Occupied voxels: {np.sum(occupied)} ({100*np.sum(occupied)/np.prod(grid_size):.2f}%)")
        voxel_color = np.zeros_like(color_sum)
        for c in range(3):
            np.divide(color_sum[:, :, :, c], color_count, out=voxel_color[:, :, :, c], where=occupied)

        # === PHASE 2: Blur voxel grid ===
        print("  Phase 2: Blurring voxel grid...")
        occupied_f = occupied.astype(np.float64)
        weight = gaussian_filter(occupied_f, sigma=blur_sigma)
        weight = np.maximum(weight, 1e-10)
        blurred_color = np.zeros_like(voxel_color)
        for c in range(3):
            blurred_color[:, :, :, c] = gaussian_filter(
                voxel_color[:, :, :, c] * occupied_f, sigma=blur_sigma
            ) / weight

        # === PHASE 3: Paint candidate nodes ===
        print(f"  Phase 3: Painting {len(poly_candidates)} nodes...")
        seen = set()
        redacted = 0

        num_poly_candidates = len(poly_candidates)
        for cand_idx, (node, is_leaf) in enumerate(poly_candidates):
            res_id = node["mesh"]["material"]["resource"]
            if res_id in seen:
                continue
            seen.add(res_id)
            print(f"    [poly ID={poly_id}] candidate {cand_idx}/{num_poly_candidates} res_id={res_id}", flush=True)

            tex_path = f"nodes/{res_id}/textures/0.jpg"
            geom_path = f"nodes/{res_id}/geometries/0.bin.gz"
            try:
                zf.getinfo(tex_path)
                zf.getinfo(geom_path)
            except KeyError:
                continue

            result = parse_geometry(zf, raw_fp, res_id)
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

            vertex_inside = np.array([
                prep_single.contains(Point(pos_world[i, 0], pos_world[i, 1]))
                for i in range(file_vc)
            ])
            tri_inside = vertex_inside.reshape(num_tri, 3).sum(axis=1)

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
                for hit_idx in tree.query(single_poly):
                    if single_poly.intersects(tri_polys_2d[hit_idx]):
                        straddle_set.add(tri_polys_idx[hit_idx])

            # Load texture — use already-modified version if available
            if tex_path in replace_paths:
                texture = Image.open(io.BytesIO(replace_paths[tex_path]))
            else:
                tex_data = raw_entry_read(zf, raw_fp, tex_path)
                texture = Image.open(io.BytesIO(tex_data))
            tex_w, tex_h = texture.size
            tex_np = np.array(texture)

            mask = Image.new("L", (tex_w, tex_h), 0)
            draw = ImageDraw.Draw(mask)
            voxel_tex = np.zeros_like(tex_np, dtype=np.float64)
            voxel_mask = np.zeros((tex_h, tex_w), dtype=np.uint8)

            for i in range(num_tri):
                n_in = tri_inside[i]
                if n_in == 0 and i not in straddle_set:
                    continue

                uv_a, uv_b, uv_c = uv0[3*i], uv0[3*i+1], uv0[3*i+2]

                if n_in == 3:
                    pts = [(uv_a[0]*tex_w, uv_a[1]*tex_h),
                           (uv_b[0]*tex_w, uv_b[1]*tex_h),
                           (uv_c[0]*tex_w, uv_c[1]*tex_h)]
                    draw.polygon(pts, fill=255)

                us = [uv_a[0], uv_b[0], uv_c[0]]
                vs = [uv_a[1], uv_b[1], uv_c[1]]
                px_min = max(0, int(min(us) * tex_w) - 1)
                px_max = min(tex_w - 1, int(max(us) * tex_w) + 1)
                py_min = max(0, int(min(vs) * tex_h) - 1)
                py_max = min(tex_h - 1, int(max(vs) * tex_h) + 1)

                wv0 = tri_pos[i, 0]
                wv1 = tri_pos[i, 1]
                wv2 = tri_pos[i, 2]

                for py in range(py_min, py_max + 1):
                    for px in range(px_min, px_max + 1):
                        u_coord = (px + 0.5) / tex_w
                        v_coord = (py + 0.5) / tex_h
                        bu, bv, bw = _bary_uv(u_coord, v_coord, uv_a, uv_b, uv_c)
                        if bu < -0.001 or bv < -0.001 or bw < -0.001:
                            continue
                        wx = bu * wv0[0] + bv * wv1[0] + bw * wv2[0]
                        wy = bu * wv0[1] + bv * wv1[1] + bw * wv2[1]
                        wz = bu * wv0[2] + bv * wv1[2] + bw * wv2[2]
                        if n_in < 3 and not prep_single.contains(Point(wx, wy)):
                            continue
                        gi = int((wx - grid_origin[0]) / voxel_size)
                        gj_idx = int((wy - grid_origin[1]) / voxel_size)
                        gk = int((wz - grid_origin[2]) / voxel_size)
                        gi = np.clip(gi, 0, grid_size[0] - 1)
                        gj_idx = np.clip(gj_idx, 0, grid_size[1] - 1)
                        gk = np.clip(gk, 0, grid_size[2] - 1)
                        voxel_tex[py, px] = blurred_color[gi, gj_idx, gk]
                        voxel_mask[py, px] = 255

            mask_np = np.maximum(np.array(mask), voxel_mask)

            if np.sum(mask_np > 0) == 0:
                continue

            # Fill full triangles that weren't per-pixel sampled
            full_only = (np.array(mask) > 0) & (voxel_mask == 0)
            if np.sum(full_only) > 0:
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
                                continue
                            if mask_np[py, px] == 0:
                                continue
                            u_coord = (px + 0.5) / tex_w
                            v_coord = (py + 0.5) / tex_h
                            bu, bv, bw = _bary_uv(u_coord, v_coord, uv_a, uv_b, uv_c)
                            if bu < -0.001 or bv < -0.001 or bw < -0.001:
                                continue
                            wx = bu * wv0[0] + bv * wv1[0] + bw * wv2[0]
                            wy = bu * wv0[1] + bv * wv1[1] + bw * wv2[1]
                            wz = bu * wv0[2] + bv * wv1[2] + bw * wv2[2]
                            gi = np.clip(int((wx - grid_origin[0]) / voxel_size), 0, grid_size[0]-1)
                            gj_idx = np.clip(int((wy - grid_origin[1]) / voxel_size), 0, grid_size[1]-1)
                            gk = np.clip(int((wz - grid_origin[2]) / voxel_size), 0, grid_size[2]-1)
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
            pad_only = (padded > 0) & (voxel_mask == 0)
            if np.sum(pad_only) > 0:
                painted = voxel_mask > 0
                k0 = painted.astype(np.float64)
                w0 = gaussian_filter(k0, sigma=1)
                reached = w0 >= 1e-10  # what one blur pass can actually support

                for c in range(3):
                    filled = gaussian_filter(voxel_tex[:, :, c] * k0, sigma=1)
                    voxel_tex[:, :, c] = np.where(
                        ~painted & reached, filled / np.maximum(w0, 1e-10), voxel_tex[:, :, c]
                    )

                # Padding dilates up to ~5.7px diagonally, but a sigma=1 blur is
                # truncated at 4px, so pixels past that got weight 0 and resolved to
                # 0/1e-10 = black. Grow the known set outward one ring per pass to
                # reach them — the source set has to expand, otherwise every pass
                # re-derives from the same pixels and never gets further out.
                known = painted | reached
                for _ in range(16):
                    if known[pad_only].all():
                        break
                    kf = known.astype(np.float64)
                    w = gaussian_filter(kf, sigma=1)
                    newly = (w > 0.02) & ~known
                    if not np.any(newly):
                        break
                    for c in range(3):
                        filled = gaussian_filter(voxel_tex[:, :, c] * kf, sigma=1)
                        voxel_tex[:, :, c] = np.where(
                            newly, filled / np.maximum(w, 1e-10), voxel_tex[:, :, c]
                        )
                    known |= newly

            mask_np = np.maximum(mask_np, padded)

            mask_bool = mask_np > 0

            # Pixels the diffusion could not reach have no coloured neighbour to spread
            # from (a UV island whose triangles never got painted, plus its padding), so
            # they are still 0 and would write as pure black. Copy the colour of the
            # nearest redacted non-black pixel instead. Sources are restricted to the
            # redacted area, so no original imagery is pulled in, and the copy is
            # un-smoothed so it cannot flatten anything around it.
            still_black = mask_bool & np.all(np.clip(voxel_tex, 0, 255) <= BLACK_THRESHOLD, axis=-1)
            source = mask_bool & ~still_black
            if np.any(still_black) and np.any(source):
                _, nidx = distance_transform_edt(~source, return_distances=True, return_indices=True)
                nidx = tuple(nidx)
                for c in range(3):
                    chan = voxel_tex[:, :, c]
                    voxel_tex[:, :, c] = np.where(still_black, chan[nidx], chan)

            tex_np[mask_bool] = np.clip(voxel_tex[mask_bool], 0, 255).astype(np.uint8)
            result = Image.fromarray(tex_np)

            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=100, subsampling=0)
            red_bytes = buf.getvalue()
            replace_paths[tex_path] = red_bytes

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

        total_redacted += redacted
        print(f"  Redacted {redacted} textures for polygon ID={poly_id}")

    print(f"\nRedacted {total_redacted} textures total")
    print(f"Total replace paths: {len(replace_paths)}")

    # === REBUILD (identical to test_all_red.py) ===
    print(f"\nRebuilding to {args.output}...")
    raw_fp.seek(0)
    entry_count = 0
    replaced = 0

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_STORED, allowZip64=True) as outz:
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
    with zipfile.ZipFile(args.output, "r") as outz:
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
    with zipfile.ZipFile(args.output, "a", allowZip64=True) as outz:
        outz.writestr("@3dtilesIndex1@", hash_data)

    print(f"  Hash table: {len(hash_entries)} entries")
    out_size = Path(args.output).stat().st_size
    print(f"\nOutput: {args.output} ({out_size / 1024**3:.1f} GB)")
    print("Done!")


if __name__ == "__main__":
    main()
