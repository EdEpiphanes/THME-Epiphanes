#!/usr/bin/env python3
"""
Render the animated THME brand banner (assets/brand/thme-banner.gif) and its static frame
(assets/brand/thme-banner.png).

A two-panel manga title spread: the mascot (assets/brand/thme-mascot.png) breathing in the left
panel; in the right panel a Calabi-Yau quintic cross-section turning in 3D while its 4D
projection angle sweeps (the surface lives in C^2), drawn with ink linework, cel tones and
screentone inside flickering focus lines, under the THME wordmark (Chakra Petch, SIL OFL 1.1).
The geometry is the same as the Studio's Live 3D manifold (src/gui/manga.rs). Brand art only:
no data is shown. Needs Pillow and NumPy (developer tools, not runtime dependencies).

  python scripts/generate_banner.py
"""

import math
import os
import sys

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    sys.exit("error: needs Pillow and NumPy (pip install pillow numpy)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, S = 1200, 420, 2  # output size and supersampling factor
FRAMES, FRAME_MS = 48, 100
OUT_W = 1000  # encoded GIF width
INK, PAPER = (10, 10, 10), (255, 255, 255)
BOLD = os.path.join(ROOT, "assets", "fonts", "ChakraPetch-Bold.ttf")
MEDIUM = os.path.join(ROOT, "assets", "fonts", "ChakraPetch-Medium.ttf")

LEFT = [(16, 16), (478, 16), (414, 404), (16, 404)]
RIGHT = [(494, 16), (1184, 16), (1184, 404), (430, 404)]
CENTER = (960, 214)


def quintic_patches(n=5, nu=16, nv=10, alpha=0.85):
    """Hanson's parametrization of z1^n + z2^n = 1, projected from C^2 to R^3."""
    x = np.linspace(-1.0, 1.0, nu)
    y = np.linspace(0.0, math.pi / 2, nv)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y
    u1 = np.cosh(z)
    u2 = np.sinh(z) / 1j
    patches = []
    for k1 in range(n):
        for k2 in range(n):
            z1 = np.exp(2j * math.pi * k1 / n) * u1 ** (2.0 / n)
            z2 = np.exp(2j * math.pi * k2 / n) * u2 ** (2.0 / n)
            pts = np.stack([z1.real, z2.real, math.cos(alpha) * z1.imag + math.sin(alpha) * z2.imag], axis=-1)
            patches.append(pts)
    return patches


def to_view(p, yaw, pitch, dist):
    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    x1 = p[..., 0] * cy - p[..., 2] * sy
    z1 = p[..., 0] * sy + p[..., 2] * cy
    y2 = p[..., 1] * cp - z1 * sp
    z2 = p[..., 1] * sp + z1 * cp + dist
    return np.stack([x1, y2, z2], axis=-1)


def draw_manifold(draw, frame, rng, center=CENTER, focal_px=330, scale=S, sparkles=True, grid=(16, 10), line=1.0):
    t = frame / FRAMES
    alpha = 0.85 + 0.55 * math.sin(2 * math.pi * t)
    yaw = 2 * math.pi * t
    patches = quintic_patches(nu=grid[0], nv=grid[1], alpha=alpha)
    focal = focal_px * scale
    cx, cy = center[0] * scale, center[1] * scale
    light = np.array([-0.45, 0.75, -0.55])
    light /= np.linalg.norm(light)
    faces = []
    for index, pts in enumerate(patches):
        v = to_view(pts[..., [0, 2, 1]] * 1.32, yaw, 0.38, 6.0)
        tone_level = (index * 7) % 5  # a family-like grey per patch
        for j in range(v.shape[0] - 1):
            for i in range(v.shape[1] - 1):
                q = [v[j, i], v[j, i + 1], v[j + 1, i + 1], v[j + 1, i]]
                n = np.cross(q[1] - q[0], q[3] - q[0])
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    continue
                n = n / norm
                if n[2] > 0:
                    n = -n
                screen = [(cx + a[0] * focal / a[2], cy - a[1] * focal / a[2]) for a in q]
                area = 0.5 * abs(sum(screen[k][0] * screen[(k + 1) % 4][1] - screen[(k + 1) % 4][0] * screen[k][1]
                                     for k in range(4)))
                if area < 2.0 * scale * scale:
                    continue
                diffuse = max(0.0, float(n @ light))
                facing = min(1.0, max(0.0, -float(n[2])))
                depth = sum(a[2] for a in q) / 4
                faces.append((depth, screen, diffuse, facing, tone_level))
    faces.sort(key=lambda f: -f[0])
    for depth, screen, diffuse, facing, tone_level in faces:
        if diffuse > 0.72 or facing < 0.22:
            fill, dots = PAPER, 0
        elif diffuse > 0.32:
            grey = 236 - 16 * tone_level
            fill, dots = (grey, grey, grey), 1 if tone_level >= 2 else 0
        else:
            grey = 120 - 12 * tone_level
            fill, dots = (grey, grey, grey), 2
        draw.polygon(screen, fill=fill)
        for k in range(4):
            draw.line([screen[k], screen[(k + 1) % 4]], fill=INK, width=max(1, round(scale * line)))
        mx = sum(p[0] for p in screen) / 4
        my = sum(p[1] for p in screen) / 4
        if dots:
            r = (0.9 if dots == 1 else 1.3) * scale
            draw.ellipse([mx - r, my - r, mx + r, my + r], fill=INK)
        if sparkles and diffuse > 0.96 and facing > 0.6 and rng.random() < 0.35:
            sparkle(draw, mx, my, 5 * scale)


def sparkle(draw, x, y, r):
    """Four-point manga sparkle."""
    draw.polygon([(x, y - r), (x + r * 0.22, y), (x, y + r), (x - r * 0.22, y)], fill=PAPER, outline=INK)
    draw.polygon([(x - r, y), (x, y + r * 0.22), (x + r, y), (x, y - r * 0.22)], fill=PAPER, outline=INK)


def focus_lines(draw, rng, inner, outer):
    """Concentration lines toward the manifold; redrawn every few frames so they flicker."""
    cx, cy = CENTER[0] * S, CENTER[1] * S
    for _ in range(150):
        a = rng.uniform(0, 2 * math.pi)
        r0 = inner * rng.uniform(1.0, 1.35)
        width = rng.uniform(0.004, 0.016)
        p0 = (cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        p1 = (cx + math.cos(a - width) * outer, cy + math.sin(a - width) * outer)
        p2 = (cx + math.cos(a + width) * outer, cy + math.sin(a + width) * outer)
        draw.polygon([p0, p1, p2], fill=255)


def halftone(draw, box, spacing, max_r, direction="down"):
    x0, y0, x1, y1 = box
    y = y0
    row = 0
    while y < y1:
        x = x0 + (spacing / 2 if row % 2 else 0)
        while x < x1:
            t = (y - y0) / max(1, y1 - y0) if direction == "down" else (x - x0) / max(1, x1 - x0)
            r = max_r * max(0.0, (t - 0.35) / 0.65)
            if r > 0.4:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=INK)
            x += spacing
        y += spacing
        row += 1


def mask_for(poly):
    mask = Image.new("L", (W * S, H * S), 0)
    ImageDraw.Draw(mask).polygon([(x * S, y * S) for x, y in poly], fill=255)
    return mask


def scaled(poly):
    return [(x * S, y * S) for x, y in poly]


def main():
    mascot_src = Image.open(os.path.join(ROOT, "assets", "brand", "thme-mascot.png")).convert("L")
    title = ImageFont.truetype(BOLD, 132 * S)
    sub = ImageFont.truetype(MEDIUM, 21 * S)
    tag = ImageFont.truetype(BOLD, 15 * S)
    left_mask, right_mask = mask_for(LEFT), mask_for(RIGHT)
    rng_lines = np.random.default_rng(7)
    rng_spark = np.random.default_rng(11)
    lines_layer = None
    frames = []
    for f in range(FRAMES):
        t = f / FRAMES
        canvas = Image.new("RGB", (W * S, H * S), (246, 246, 244))
        draw = ImageDraw.Draw(canvas)

        # Left panel: the mascot breathing (two slow breaths per loop), anchored at her chest.
        breath = math.sin(2 * math.pi * 2 * t)
        size = int(470 * S * (1.0 + 0.012 * breath))
        mascot = mascot_src.resize((size, size), Image.LANCZOS).convert("RGB")
        panel = Image.new("RGB", (W * S, H * S), PAPER)
        mx = int(250 * S - size * 0.52)
        my = int(404 * S - size * 0.93 - 2.5 * S * breath)
        panel.paste(mascot, (mx, my))
        ImageDraw.Draw(panel)
        canvas.paste(panel, (0, 0), left_mask)

        # Right panel: paper, focus lines (new every 3 frames), halftone floor, manifold.
        right = Image.new("RGB", (W * S, H * S), PAPER)
        rdraw = ImageDraw.Draw(right)
        if f % 3 == 0 or lines_layer is None:
            lines_layer = Image.new("L", (W * S, H * S), 0)
            focus_lines(ImageDraw.Draw(lines_layer), rng_lines, 185 * S, 900 * S)
        right.paste(INK, (0, 0), lines_layer)
        halftone(rdraw, (430 * S, 300 * S, 1184 * S, 404 * S), 9 * S, 2.2 * S)
        # Clear halo behind the manifold so it reads against the lines.
        rdraw.ellipse([(CENTER[0] - 178) * S, (CENTER[1] - 178) * S, (CENTER[0] + 178) * S, (CENTER[1] + 178) * S],
                      fill=PAPER)
        draw_manifold(rdraw, f, rng_spark)
        canvas.paste(right, (0, 0), right_mask)

        # Wordmark with a hard print shadow, caption box and tag.
        tx, ty = 512 * S, 36 * S
        draw.text((tx + 7 * S, ty + 7 * S), "THME", font=title, fill=INK)
        draw.text((tx, ty), "THME", font=title, fill=PAPER, stroke_width=4 * S, stroke_fill=INK)
        box = (512 * S, 206 * S, 512 * S + 250 * S, 206 * S + 92 * S)
        draw.rectangle((box[0] + 5 * S, box[1] + 5 * S, box[2] + 5 * S, box[3] + 5 * S), fill=INK)
        draw.rectangle(box, fill=PAPER, outline=INK, width=3 * S)
        draw.text((box[0] + 12 * S, box[1] + 8 * S), "TOPOLOGICAL", font=sub, fill=INK)
        draw.text((box[0] + 12 * S, box[1] + 34 * S), "HETEROGENEOUS", font=sub, fill=INK)
        draw.text((box[0] + 12 * S, box[1] + 60 * S), "MANIFOLD ENGINE", font=sub, fill=INK)
        tag_box = (512 * S, 318 * S, 512 * S + 134 * S, 318 * S + 26 * S)
        draw.rectangle(tag_box, fill=INK)
        draw.text((tag_box[0] + 10 * S, tag_box[1] + 4 * S), "BY EPIPHANES", font=tag, fill=PAPER)

        # Panel frames and the outer border.
        draw.polygon(scaled(LEFT), outline=INK, width=5 * S)
        draw.polygon(scaled(RIGHT), outline=INK, width=5 * S)

        frame = canvas.resize((OUT_W, round(H * OUT_W / W)), Image.LANCZOS).convert("L")
        frames.append(frame)

    out_dir = os.path.join(ROOT, "assets", "brand")
    frames[0].save(os.path.join(out_dir, "thme-banner.png"), optimize=True)
    palette = frames[FRAMES // 2].convert("P", palette=Image.ADAPTIVE, colors=20)
    encoded = [f.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif = os.path.join(out_dir, "thme-banner.gif")
    encoded[0].save(gif, save_all=True, append_images=encoded[1:], duration=FRAME_MS, loop=0, optimize=True,
                    disposal=1)
    # Logo mark: the manifold alone in an ink panel (app-icon and avatar sized).
    mark_s = 4
    mark = Image.new("RGB", (256 * mark_s, 256 * mark_s), PAPER)
    mdraw = ImageDraw.Draw(mark)
    halftone(mdraw, (0, 176 * mark_s, 256 * mark_s, 256 * mark_s), 10 * mark_s, 1.6 * mark_s)
    draw_manifold(mdraw, FRAMES // 8, np.random.default_rng(3), center=(128, 132), focal_px=205,
                  scale=mark_s, sparkles=False, grid=(9, 6), line=0.6)
    mdraw.rectangle((4 * mark_s, 4 * mark_s, 251 * mark_s, 251 * mark_s), outline=INK, width=8 * mark_s)
    mark.resize((256, 256), Image.LANCZOS).convert("L").save(os.path.join(out_dir, "thme-mark-256.png"), optimize=True)
    print(f"wrote assets/brand/thme-banner.gif ({len(frames)} frames, {os.path.getsize(gif) / 1024:.0f} KiB) "
          f"and assets/brand/thme-banner.png, thme-mark-256.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
