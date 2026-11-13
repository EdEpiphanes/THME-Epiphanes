# SPDX-License-Identifier: AGPL-3.0-only
"""Renders a Calabi-Yau quintic slice (Hanson's construction) as a flat-shaded SVG mark.

The Fermat quintic slice z1^5 + z2^5 = 1 is covered by 25 patches. Patch (k1, k2) is
  z1 = e^(2 pi i k1/5) cosh(w)^(2/5),  z2 = e^(2 pi i k2/5) e^(i pi/5) sinh(w)^(2/5),
with w = x + i y, x in [-a, a], y in [0, pi/2]. A point is drawn at
(Re z1, Re z2, cos(t) Im z1 + sin(t) Im z2). Patches are coloured by representation family.
Usage: python -I scripts/generate_logo.py assets/brand/calabi-yau.svg
"""
import cmath
import math
import sys

N, A, STEPS = 5, 1.0, 8
FAMILIES = [((0x00, 0xB4, 0xD8)), ((0x10, 0xB9, 0x81)), ((0xF5, 0x9E, 0x0B))]


def patch_point(k1, k2, x, y, t=math.pi / 4):
    w = complex(x, y)
    z1 = cmath.exp(2j * math.pi * k1 / N) * cmath.cosh(w) ** (2 / N)
    z2 = cmath.exp(2j * math.pi * k2 / N) * cmath.exp(1j * math.pi / N) * cmath.sinh(w) ** (2 / N)
    return (z1.real, z2.real, math.cos(t) * z1.imag + math.sin(t) * z2.imag)


def rotate(p, yaw, pitch):
    x, y, z = p
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cp, sp = math.cos(pitch), math.sin(pitch)
    y, z = y * cp - z * sp, y * sp + z * cp
    return x, y, z


def main():
    out = sys.argv[1]
    yaw, pitch = 0.55, -0.35
    light = (0.35, 0.55, 0.76)
    quads = []
    for k1 in range(N):
        for k2 in range(N):
            family = FAMILIES[(k1 + 2 * k2) % 3]
            grid = [[rotate(patch_point(k1, k2, -A + 2 * A * i / STEPS, (math.pi / 2) * j / STEPS), yaw, pitch)
                     for j in range(STEPS + 1)] for i in range(STEPS + 1)]
            for i in range(STEPS):
                for j in range(STEPS):
                    p = [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]]
                    ux, uy, uz = (p[1][k] - p[0][k] for k in range(3))
                    vx, vy, vz = (p[3][k] - p[0][k] for k in range(3))
                    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                    norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                    shade = abs(nx * light[0] + ny * light[1] + nz * light[2]) / norm
                    depth = sum(q[2] for q in p) / 4
                    quads.append((depth, p, family, shade))
    xs = [q[0] for _, pts, _, _ in quads for q in pts]
    ys = [q[1] for _, pts, _, _ in quads for q in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    scale, size = 440 / span, 512
    ox = size / 2 - scale * (max(xs) + min(xs)) / 2
    oy = size / 2 + scale * (max(ys) + min(ys)) / 2
    quads.sort(key=lambda q: q[0])  # far (small z) first
    parts = []
    for _, pts, (r, g, b), shade in quads:
        k = 0.45 + 0.55 * shade
        color = f"#{int(r * k):02x}{int(g * k):02x}{int(b * k):02x}"
        d = " ".join(f"{ox + scale * x:.1f},{oy - scale * y:.1f}" for x, y, _ in pts)
        parts.append(f'<polygon points="{d}" fill="{color}" stroke="{color}" stroke-width="0.6"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">\n'
           f'<title>THME: Calabi-Yau quintic slice in three representation families</title>\n'
           + "\n".join(parts) + "\n</svg>\n")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    print(f"{len(quads)} faces, {len(svg) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
