#!/usr/bin/env python3
"""Use the WORKING rebuild code (from test_all_red.py) but with UV-masked textures
instead of solid red. If this works → redact.py's rebuild has a bug.
If this fails → the texture content itself is the problem."""

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
from shapely.geometry import Polygon, Point, box, shape
from shapely.prepared import prep

SLPK = "/home/sbk-usr/mesh-redaction/Stockholm_3D_2025_North.slpk"
OUTPUT = "/home/sbk-usr/mesh-redaction/test_uvred_clean.slpk"


def _bary_uv(px, py, uv_a, uv_b, uv_c):
    """Barycentric coords of (px,py) within UV triangle (uv_a, uv_b, uv_c).
    UV space is always a well-conditioned 2D triangle."""
    d00 = (uv_b[0]-uv_a[0])**2 + (uv_b[1]-uv_a[1])**2
    d01 = (uv_b[0]-uv_a[0])*(uv_c[0]-uv_a[0]) + (uv_b[1]-uv_a[1])*(uv_c[1]-uv_a[1])
    d11 = (uv_c[0]-uv_a[0])**2 + (uv_c[1]-uv_a[1])**2
    d20 = (px-uv_a[0])*(uv_b[0]-uv_a[0]) + (py-uv_a[1])*(uv_b[1]-uv_a[1])
    d21 = (px-uv_a[0])*(uv_c[0]-uv_a[0]) + (py-uv_a[1])*(uv_c[1]-uv_a[1])
    denom = d00*d11 - d01*d01
    if abs(denom) < 1e-12:
        return -1, -1, -1  # degenerate UV triangle
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    u = 1 - v - w
    return u, v, w


def main():
    print("=== UV-Masked Red (Clean Rebuild) ===")

    zipfile.ZipExtFile._update_crc = lambda self, data: None
    zf = zipfile.ZipFile(SLPK, "r")

    # Load layer info
    layer = json.loads(gzip.decompress(zf.read("3dSceneLayer.json.gz")))
    has_color = "color" in layer["store"]["defaultGeometrySchema"]["vertexAttributes"]

    # Load polygon
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3011", always_xy=True)
    with open("region.geojson") as f:
        gj = json.load(f)
    geom = shape(gj["features"][0]["geometry"])
    coords = [transformer.transform(lon, lat) for lon, lat in geom.exterior.coords]
    poly = Polygon(coords)
    print(f"Polygon bounds: {poly.bounds}")

    # Find ALL candidates from node pages
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
            node_box = box(cx - hx, cy - hy, cx + hx, cy + hy)
            if node_box.intersects(poly):
                candidates.append(node)

    print(f"Candidates: {len(candidates)}")

    # Process each candidate — UV masking with red fill
    replace_paths = {}
    seen = set()
    redacted = 0

    for node in candidates:
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

        # Parse geometry (full header_vc)
        gdata = gzip.decompress(zf.read(geom_path))
        if len(gdata) < 8:
            continue
        file_vc = struct.unpack_from("<I", gdata, 0)[0]
        if file_vc == 0:
            continue

        offset = 8
        pos_size = file_vc * 3 * 4
        if offset + pos_size > len(gdata):
            continue
        positions = np.frombuffer(gdata, dtype=np.float32, count=file_vc * 3, offset=offset).reshape(-1, 3)
        offset += pos_size
        offset += file_vc * 3 * 4  # normals
        uv_size = file_vc * 2 * 4
        if offset + uv_size > len(gdata):
            continue
        uv0 = np.frombuffer(gdata, dtype=np.float32, count=file_vc * 2, offset=offset).reshape(-1, 2)

        # World positions
        obb_center = node["obb"]["center"]
        pos_world = positions.copy()
        pos_world[:, 0] += obb_center[0]
        pos_world[:, 1] += obb_center[1]
        pos_world[:, 2] += obb_center[2]

        # Find triangles to mask
        num_tri = file_vc // 3
        tri_pos = pos_world.reshape(num_tri, 3, 3)
        prep_p = prep(poly)

        # Vertex containment: vectorized check for all vertices
        vertex_inside = np.array([
            prep_p.contains(Point(pos_world[i, 0], pos_world[i, 1]))
            for i in range(file_vc)
        ])
        # Per-triangle: how many vertices inside?
        tri_inside = vertex_inside.reshape(num_tri, 3).sum(axis=1)

        # Load texture early — needed for per-pixel boundary approach
        tex_data = zf.read(tex_path)
        texture = Image.open(io.BytesIO(tex_data))
        tex_w, tex_h = texture.size

        mask = Image.new("L", (tex_w, tex_h), 0)
        draw = ImageDraw.Draw(mask)
        mask_np = np.zeros((tex_h, tex_w), dtype=np.uint8)

        n_full = 0
        n_boundary = 0

        # Also find triangles with 0 vertices inside but that still intersect
        # (large triangles straddling the polygon boundary)
        from shapely.strtree import STRtree
        straddle_set = set()
        tri_polys_2d = []
        tri_polys_idx = []
        for i in range(num_tri):
            if tri_inside[i] > 0:
                continue  # already handled
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

        for i in range(num_tri):
            n_in = tri_inside[i]
            if n_in == 0 and i not in straddle_set:
                continue

            uv_a, uv_b, uv_c = uv0[3*i], uv0[3*i+1], uv0[3*i+2]

            if n_in == 3:
                # All vertices inside polygon → paint full UV triangle
                pts = [(uv_a[0]*tex_w, uv_a[1]*tex_h),
                       (uv_b[0]*tex_w, uv_b[1]*tex_h),
                       (uv_c[0]*tex_w, uv_c[1]*tex_h)]
                draw.polygon(pts, fill=255)
                n_full += 1
            else:
                # Boundary triangle: per-pixel reverse mapping (UV → world)
                # Compute bounding box in pixel space
                us = [uv_a[0], uv_b[0], uv_c[0]]
                vs = [uv_a[1], uv_b[1], uv_c[1]]
                px_min = max(0, int(min(us) * tex_w) - 1)
                px_max = min(tex_w - 1, int(max(us) * tex_w) + 1)
                py_min = max(0, int(min(vs) * tex_h) - 1)
                py_max = min(tex_h - 1, int(max(vs) * tex_h) + 1)

                # World XY vertices for this triangle
                wv0 = tri_pos[i, 0, :2]
                wv1 = tri_pos[i, 1, :2]
                wv2 = tri_pos[i, 2, :2]

                for py in range(py_min, py_max + 1):
                    for px in range(px_min, px_max + 1):
                        # Pixel center in UV space
                        u_coord = (px + 0.5) / tex_w
                        v_coord = (py + 0.5) / tex_h

                        # Barycentric in UV space (always well-conditioned)
                        bu, bv, bw = _bary_uv(u_coord, v_coord, uv_a, uv_b, uv_c)
                        if bu < -0.001 or bv < -0.001 or bw < -0.001:
                            continue  # outside UV triangle

                        # Map to world XY using same barycentric weights
                        wx = bu * wv0[0] + bv * wv1[0] + bw * wv2[0]
                        wy = bu * wv0[1] + bv * wv1[1] + bw * wv2[1]

                        # Test against polygon
                        if prep_p.contains(Point(wx, wy)):
                            mask_np[py, px] = 255

                n_boundary += 1

        # Merge rasterized full triangles with per-pixel boundary mask
        mask_np = np.maximum(mask_np, np.array(mask))

        if n_full + n_boundary == 0:
            continue

        # UV padding: extend mask into dead space by 4px for bilinear + mipmap sampling.
        # Only expand INTO dead space (pixels not covered by ANY UV triangle),
        # never into adjacent UV islands.
        from scipy.ndimage import binary_dilation
        all_uv_mask = np.array(mask)  # all UV triangles (full node, not just polygon)
        # Rebuild all-triangles mask to identify dead space
        all_tri_mask = Image.new("L", (tex_w, tex_h), 0)
        all_tri_draw = ImageDraw.Draw(all_tri_mask)
        for ti in range(num_tri):
            a, b, c = uv0[3*ti], uv0[3*ti+1], uv0[3*ti+2]
            pts = [(a[0]*tex_w, a[1]*tex_h), (b[0]*tex_w, b[1]*tex_h), (c[0]*tex_w, c[1]*tex_h)]
            all_tri_draw.polygon(pts, fill=255)
        dead_space = np.array(all_tri_mask) == 0

        # Dilate the mask, but only allow expansion into dead space
        padded = binary_dilation(mask_np > 0, iterations=4).astype(np.uint8) * 255
        padded[~dead_space & (mask_np == 0)] = 0  # don't bleed into other UV islands
        mask_np = np.maximum(mask_np, padded)

        print(f"  res={res_id}: {n_full} full + {n_boundary} boundary triangles, "
              f"{np.sum(mask_np > 0)} masked px")

        # Apply pixelation + blur to masked region
        tex_np = np.array(texture)
        mask_bool = mask_np > 0

        # Pixelate: downscale then upscale to create blocky effect
        pixelate_factor = 16
        small_w, small_h = max(1, tex_w // pixelate_factor), max(1, tex_h // pixelate_factor)
        pixelated = texture.resize((small_w, small_h), Image.NEAREST).resize((tex_w, tex_h), Image.NEAREST)
        pix_np = np.array(pixelated)

        # Gaussian blur the pixelated result for softer appearance
        from PIL import ImageFilter
        blurred = Image.fromarray(pix_np).filter(ImageFilter.GaussianBlur(radius=8))
        blur_np = np.array(blurred)

        # Composite: blurred pixelation in masked region, original elsewhere
        tex_np[mask_bool] = blur_np[mask_bool]
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
