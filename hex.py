#!/usr/bin/env python3
# hex_thirds_hexfootprint_pointy_fit.py
#
# Pointy-top hexagon footprint with S hexes per side.
# User specifies final image width/height in pixels; script chooses hex size
# so the shape fits inside exactly and is centered (no distortion).
# Each hex is split into thirds by center->midpoints on:
#   top side, bottom-right side, bottom-left side.
# Optional per-hex colors via CSV: q,r,color1,color2,color3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------------------- Geometry (pointy-top) -----------------------

def hex_vertices_pointy(center, r):
    """Return 6 vertices of a pointy-top hexagon CCW from angle 30°."""
    cx, cy = center
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((cx + r*np.cos(angles), cy + r*np.sin(angles)))

def side_midpoint(verts, i, j):
    p = verts[i % 6]; q = verts[j % 6]
    return (p + q) / 2.0

def axial_to_pixel_pointy(q, r, size):
    """
    Pointy-top axial -> pixel (cartesian).
      x = size * sqrt(3) * (q + r/2)
      y = size * 3/2 * r
    """
    x = size * math.sqrt(3) * (q + r/2.0)
    y = size * (1.5 * r)
    return (x, y)

def hex_region_axial(side):
    """
    Axial coordinates (q,r) in a hex-shaped region with 'side' tiles per edge.
    Radius R = side - 1, constraint |q|<=R, |r|<=R, |q+r|<=R.
    """
    R = side - 1
    coords = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if abs(q + r) <= R and abs(q) <= R and abs(r) <= R:
                coords.append((q, r))
    return coords

# ----------------------- Draw one hex (thirds) -----------------------

def draw_hex_with_thirds_pointy(ax, center, size, colors,
                                edge_color="black", edge_lw=1.0, alpha=1.0):
    verts = hex_vertices_pointy(center, size)
    cx, cy = center
    v0, v1, v2, v3, v4, v5 = verts

    # Midpoints on designated sides
    m_top = side_midpoint(verts, 0, 1)  # top side
    m_br  = side_midpoint(verts, 4, 5)  # bottom-right side
    m_bl  = side_midpoint(verts, 2, 3)  # bottom-left side

    def fill_poly(points, color):
        poly = np.array(points)
        ax.fill(poly[:,0], poly[:,1], facecolor=color, edgecolor="none", alpha=alpha)

    # Third 1: top midpoint -> ... -> bottom-right midpoint (CCW along boundary)
    fill_poly([center, m_top, v1, v2, v3, v4, m_br, center], colors[0])
    # Third 2: bottom-right midpoint -> ... -> bottom-left midpoint
    fill_poly([center, m_br, v5, v0, m_top, v1, v2, m_bl, center], colors[1])
    # Third 3: bottom-left midpoint -> ... -> top midpoint
    fill_poly([center, m_bl, v3, v4, m_br, v5, v0, m_top, center], colors[2])

    # Outline + center lines
    hex_path = np.vstack([verts, verts[0]])
    ax.plot(hex_path[:,0], hex_path[:,1], color=edge_color, linewidth=edge_lw)
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

# ----------------------- Fit-to-pixels logic -----------------------

def compute_unit_bbox(side):
    """
    Compute the exact polygonal bounding box (min/max x/y) of the hex footprint
    at unit hex size (size=1), using the vertices of every hex.
    """
    coords = hex_region_axial(side)
    all_x = []
    all_y = []
    for q, r in coords:
        cx, cy = axial_to_pixel_pointy(q, r, 1.0)
        verts = hex_vertices_pointy((cx, cy), 1.0)
        all_x.extend(verts[:,0])
        all_y.extend(verts[:,1])
    return (min(all_x), max(all_x), min(all_y), max(all_y))

def plot_hex_footprint_pointy_fit(side, width_px, height_px,
                                  edge_color="black", edge_lw=1.0,
                                  default_c1="#f4d03f", default_c2="#5dade2", default_c3="#58d68d",
                                  alpha=1.0, rotate180=False, csv_map=None):
    # 1) Compute unit-size bounds (size=1)
    minx, maxx, miny, maxy = compute_unit_bbox(side)
    spanx = maxx - minx
    spany = maxy - miny

    # 2) Compute uniform scale so footprint fits inside requested pixels
    scale_x = width_px / spanx
    scale_y = height_px / spany
    scale   = min(scale_x, scale_y)   # keep hexes regular

    # 3) Center the footprint in the target canvas
    offset_x = (width_px  - spanx*scale) / 2.0
    offset_y = (height_px - spany*scale) / 2.0

    # 4) Prepare drawing
    figdpi = 100  # controls figsize mapping to pixels
    fig, ax = plt.subplots(figsize=(width_px/figdpi, height_px/figdpi), dpi=figdpi)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(0, width_px); ax.set_ylim(0, height_px)
    fig.subplots_adjust(0, 0, 1, 1)

    colors_map = load_color_map(csv_map)
    coords = hex_region_axial(side)

    # 5) Draw every hex with scaled centers and scaled size
    for q, r in coords:
        ux, uy = axial_to_pixel_pointy(q, r, 1.0)          # unit coords
        px = (ux - minx) * scale + offset_x                # pixel coords
        py = (uy - miny) * scale + offset_y
        colors = colors_map.get((q, r), (default_c1, default_c2, default_c3))
        draw_hex_with_thirds_pointy(ax, (px, py), scale, colors,
                                    edge_color=edge_color, edge_lw=edge_lw, alpha=alpha)

    # Optional whole-image 180° rotation (by flipping axes)
    if rotate180:
        ax.set_xlim(width_px, 0)
        ax.set_ylim(height_px, 0)

    return fig

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pointy-top hex footprint; fits exactly inside a W×H pixel image, with S hexes per side."
    )
    ap.add_argument("--side", type=int, default=4,
                    help="number of hexes per edge of the overall hex footprint (default: 4)")
    ap.add_argument("--width", type=int, default=800, help="output width in pixels")
    ap.add_argument("--height", type=int, default=800, help="output height in pixels")
    ap.add_argument("--linewidth", type=float, default=1.0, help="outline stroke width")
    ap.add_argument("--edgecolor", type=str, default="black", help="outline color")
    ap.add_argument("--alpha", type=float, default=1.0, help="fill alpha 0..1")
    ap.add_argument("--default1", type=str, default="#f4d03f", help="default color for third 1")
    ap.add_argument("--default2", type=str, default="#5dade2", help="default color for third 2")
    ap.add_argument("--default3", type=str, default="#58d68d", help="default color for third 3")
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV with per-hex colors: q,r,color1,color2,color3")
    ap.add_argument("--rotate180", action="store_true", help="rotate the drawing 180°")
    ap.add_argument("--output", type=str, default="hex_pointy_fit.png",
                    help="output file name (.png, .svg, .pdf, etc.)")
    args = ap.parse_args()

    fig = plot_hex_footprint_pointy_fit(
        side=args.side, width_px=args.width, height_px=args.height,
        edge_color=args.edgecolor, edge_lw=args.linewidth,
        default_c1=args.default1, default_c2=args.default2, default_c3=args.default3,
        alpha=args.alpha, rotate180=args.rotate180, csv_map=args.csv
    )

    out = Path(args.output)
    # Note: bbox_inches=None and pad_inches=0 preserve exact pixel dimensions.
    fig.savefig(out, dpi=fig.dpi, bbox_inches=None, pad_inches=0)
    print(f"Saved to {out.resolve()}")

if __name__ == "__main__":
    main()

