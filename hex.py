#!/usr/bin/env python3
# hex_thirds_hexfootprint_pointy_oriented.py
#
# Pointy-top hexagon footprint with S hexes per side.
# User specifies final image width/height in pixels; script chooses hex size
# so the footprint fits inside and is centered (no distortion).
# Each hex is split into thirds by center->midpoints on:
#   top side, bottom-right side, bottom-left side.
# Per-hex colors via CSV: q,r,color1,color2,color3
# Drawn directly in the requested orientation (default 90° CCW), no post-rotation.

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------------------- Geometry (pointy-top) -----------------------

def hex_vertices_pointy(center, r, base_angle_deg=90):
    """
    Return 6 vertices of a pointy-top hexagon, CCW.
    Standard pointy-top vertex angles start at 30°, so we apply a base offset.
    Default base_angle_deg=90 draws the whole layout already rotated 90° CCW.
    """
    cx, cy = center
    base = math.radians(base_angle_deg)
    # canonical angles for pointy-top: [30, 90, 150, 210, 270, 330]
    angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + base
    return np.column_stack((cx + r*np.cos(angles), cy + r*np.sin(angles)))

def side_midpoint(verts, i, j):
    p = verts[i % 6]; q = verts[j % 6]
    return (p + q) / 2.0

def axial_to_pixel_pointy(q, r, size, base_angle_deg=90):
    """
    Axial -> pixel for pointy-top hexes, then apply base orientation.
      canonical (unrotated):
        x = size * sqrt(3) * (q + r/2)
        y = size * 3/2 * r
      then rotate by base_angle_deg CCW around origin
    """
    x = size * math.sqrt(3) * (q + r/2.0)
    y = size * (1.5 * r)
    # rotate by base angle
    a = math.radians(base_angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xr = ca*x - sa*y
    yr = sa*x + ca*y
    return (xr, yr)

def hex_region_axial(side):
    """Axial coordinates (q,r) in a hex-shaped region with 'side' tiles per edge."""
    R = side - 1
    coords = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if abs(q + r) <= R and abs(q) <= R and abs(r) <= R:
                coords.append((q, r))
    return coords

# ----------------------- Draw one hex (thirds) -----------------------

def draw_hex_with_thirds_pointy(ax, center, size, colors,
                                edge_color="black", edge_lw=1.0, alpha=1.0,
                                base_angle_deg=90):
    verts = hex_vertices_pointy(center, size, base_angle_deg)
    cx, cy = center
    v0, v1, v2, v3, v4, v5 = verts

    # Midpoints on designated sides (indices track the oriented vertices)
    m_top = side_midpoint(verts, 0, 1)  # "top" side in this oriented frame
    m_br  = side_midpoint(verts, 4, 5)
    m_bl  = side_midpoint(verts, 2, 3)




    def rotate_point(point, center, angle_deg):
        # Rotate a point around center by angle_deg
        angle_rad = np.deg2rad(angle_deg)
        px, py = point
        cx, cy = center
        dx, dy = px - cx, py - cy
        qx = cx + dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        qy = cy + dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        return np.array([qx, qy])

    def label_third(third_idx, color_name):
        # Place label well inside each third, away from separating lines
        if third_idx == 1:
            outer = np.array([v1, v2, v3, v4])
        elif third_idx == 2:
            outer = np.array([v5, v0, m_top, v1, v2])
        elif third_idx == 3:
            outer = np.array([v3, v4, m_br, v5, v0])
        else:
            return
        centroid_outer = np.mean(outer, axis=0)
        pos = 0.7 * centroid_outer + 0.3 * np.array(center)
        # Rotate label position by 60 degrees around center
        pos_rot = rotate_point(pos, center, 60)
        print(f"Third {third_idx}: color={color_name}, label_pos={pos_rot}")
        ax.text(pos_rot[0], pos_rot[1], str(third_idx), color="black", fontsize=size*0.25, ha="center", va="center", weight="bold")

    # Thirds with improved label placement
    # Log vertices
    print(f"Hex center: {center}")
    print(f"Vertices:")
    for i, v in enumerate([v0, v1, v2, v3, v4, v5]):
        print(f"  v{i}: {v}")
    # Log midpoints
    print(f"Midpoints:")
    print(f"  m_top: {m_top}")
    print(f"  m_br: {m_br}")
    print(f"  m_bl: {m_bl}")

    # Redefine thirds using midpoints so edges intercept hex sides at 90 deg
    # Third 1: between m_br and m_top (blue)
    third1 = [center, m_br, v5, v0, m_top, center]
    # Third 2: between m_top and m_bl (red)
    third2 = [center, m_top, v1, v2, m_bl, center]
    # Third 3: between m_bl and m_br (green)
    third3 = [center, m_bl, v3, v4, m_br, center]

    print("Third polygons:")
    print(f"  third1: {[tuple(p) for p in third1]}")
    print(f"  third2: {[tuple(p) for p in third2]}")
    print(f"  third3: {[tuple(p) for p in third3]}")

    print("Drawing thirds for single hexagon:")
    print("Drawing thirds for single hexagon:")
    print(f"  third1 color: blue")
    ax.fill(np.array(third1)[:,0], np.array(third1)[:,1], facecolor="blue", edgecolor="none", alpha=alpha)
    label_third(1, "blue")
    print(f"  third2 color: red")
    ax.fill(np.array(third2)[:,0], np.array(third2)[:,1], facecolor="red", edgecolor="none", alpha=alpha)
    label_third(2, "red")
    print(f"  third3 color: green")
    ax.fill(np.array(third3)[:,0], np.array(third3)[:,1], facecolor="green", edgecolor="none", alpha=alpha)
    label_third(3, "green")

    # Outline + center lines
    poly = np.vstack([verts, verts[0]])
    ax.plot(poly[:,0], poly[:,1], color=edge_color, linewidth=edge_lw)
    for (mx, my) in [m_top, m_br, m_bl]:
        ax.plot([cx, mx], [cy, my], color=edge_color, linewidth=edge_lw)

# ----------------------- CSV color map (per-hex) -----------------------

def load_color_map(csv_path):
    """
    CSV header: q,r,color1,color2,color3
    Returns dict[(q,r)] = (c1, c2, c3)
    """
    mapping = {}
    if not csv_path:
        return mapping
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = int(row["q"]); r = int(row["r"])
            c1 = row["color1"].strip()
            c2 = row["color2"].strip()
            c3 = row["color3"].strip()
            mapping[(q, r)] = (c1, c2, c3)
    return mapping

# ----------------------- Fit-to-pixels logic (drawn oriented) -----------------------

def compute_unit_bbox(side, base_angle_deg):
    """
    Compute bounding box of the entire footprint at unit size (size=1)
    using oriented centers and oriented hex vertices.
    """
    coords = hex_region_axial(side)
    all_x = []; all_y = []
    for q, r in coords:
        cx, cy = axial_to_pixel_pointy(q, r, 1.0, base_angle_deg)
        verts = hex_vertices_pointy((cx, cy), 1.0, base_angle_deg)
        all_x.extend(verts[:,0]); all_y.extend(verts[:,1])
    return min(all_x), max(all_x), min(all_y), max(all_y)

def plot_hex_footprint_pointy_oriented(
    side, width_px, height_px,
    edge_color="black", edge_lw=1.0,
    default_c1="#f4d03f", default_c2="#5dade2", default_c3="#58d68d",
    alpha=1.0, orientation_deg=90, csv_map=None
):
    # 1) Compute unit-size bounds in the oriented frame
    minx, maxx, miny, maxy = compute_unit_bbox(side, orientation_deg)
    spanx, spany = (maxx - minx), (maxy - miny)

    # 2) Uniform scale to fit inside requested pixel area (contain)
    scale = min(width_px / spanx, height_px / spany)

    # 3) Centering offsets
    offset_x = (width_px  - spanx*scale) / 2.0
    offset_y = (height_px - spany*scale) / 2.0

    # 4) Prepare figure/canvas at exact pixels
    figdpi = 100
    fig, ax = plt.subplots(figsize=(width_px/figdpi, height_px/figdpi), dpi=figdpi)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(0, width_px); ax.set_ylim(0, height_px)
    fig.subplots_adjust(0, 0, 1, 1)

    colors_map = load_color_map(csv_map)
    coords = hex_region_axial(side)

    # 5) Draw each hex directly in the oriented coordinates
    for q, r in coords:
        cx_u, cy_u = axial_to_pixel_pointy(q, r, 1.0, orientation_deg)
        px = (cx_u - minx) * scale + offset_x
        py = (cy_u - miny) * scale + offset_y
        colors = colors_map.get((q, r), (default_c1, default_c2, default_c3))
        draw_hex_with_thirds_pointy(ax, (px, py), scale, colors,
                                    edge_color=edge_color, edge_lw=edge_lw,
                                    alpha=alpha, base_angle_deg=orientation_deg)
    return fig

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pointy-top hex footprint; fits W×H px, S hexes per side, drawn in-place at desired orientation."
    )
    ap.add_argument("--side", type=int, default=4,
                    help="number of hexes per edge of the overall hex footprint (default: 4)")
    ap.add_argument("--width", type=int, default=800, help="output width in pixels")
    ap.add_argument("--height", type=int, default=800, help="output height in pixels")
    ap.add_argument("--linewidth", type=float, default=1.0, help="outline stroke width")
    ap.add_argument("--edgecolor", type=str, default="black", help="outline color")
    ap.add_argument("--alpha", type=float, default=1.0, help="fill alpha 0..1")

    # Default orientation is 90° CCW so you don't need an extra rotation step.
    ap.add_argument("--orientation", type=float, default=90.0,
                    help="base drawing orientation in degrees CCW (default: 90)")

    ap.add_argument("--default1", type=str, default="#f4d03f", help="default color for third 1")
    ap.add_argument("--default2", type=str, default="#5dade2", help="default color for third 2")
    ap.add_argument("--default3", type=str, default="#58d68d", help="default color for third 3")

    ap.add_argument("--csv", type=str, default=None,
                    help="CSV with per-hex colors: q,r,color1,color2,color3")

    ap.add_argument("--output", type=str, default="hex_pointy_oriented.png",
                    help="output file name (.png, .svg, .pdf, etc.)")
    args = ap.parse_args()

    fig = plot_hex_footprint_pointy_oriented(
        side=args.side, width_px=args.width, height_px=args.height,
        edge_color=args.edgecolor, edge_lw=args.linewidth,
        default_c1=args.default1, default_c2=args.default2, default_c3=args.default3,
        alpha=args.alpha, orientation_deg=args.orientation, csv_map=args.csv
    )

    out = Path(args.output)
    fig.savefig(out, dpi=fig.dpi, bbox_inches=None, pad_inches=0)
    print(f"Saved to {out.resolve()}")

if __name__ == "__main__":
    main()
