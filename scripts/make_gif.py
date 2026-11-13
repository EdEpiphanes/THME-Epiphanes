#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Turn a THME Studio frame recording (rec-000.bmp, rec-001.bmp, ...) into a README GIF.

The frames are real application renders recorded by `THME_GUI_RECORD` (see
scripts/record_gif.ps1); this script only crops, scales and encodes them.
Needs Pillow (a developer tool here, not a runtime dependency).

  python scripts/make_gif.py --frames target/studio-record --out assets/gifs/live-3d-manifold.gif \
      --crop 0,320,1060,840 --width 900 --fps 15
"""

import argparse
import glob
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("error: Pillow is required (pip install pillow)")


def main():
    parser = argparse.ArgumentParser(description="Encode a Studio frame recording as a GIF")
    parser.add_argument("--frames", required=True, help="directory with rec-*.bmp frames")
    parser.add_argument("--out", required=True, help="output .gif path")
    parser.add_argument("--crop", default=None, help="x0,y0,x1,y1 in source pixels")
    parser.add_argument("--width", type=int, default=900, help="output width in pixels")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--colors", type=int, default=64, help="palette size (manga panels need few)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "rec-*.bmp")))
    if not paths:
        sys.exit(f"error: no rec-*.bmp frames in {args.frames}")
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None

    frames = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if crop:
                image = image.crop(crop)
            height = round(image.height * args.width / image.width)
            image = image.resize((args.width, height), Image.LANCZOS)
            frames.append(image)

    # One shared palette keeps flat manga tones stable from frame to frame (no flicker).
    palette = frames[len(frames) // 2].quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    encoded = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    encoded[0].save(
        args.out,
        save_all=True,
        append_images=encoded[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}: {len(encoded)} frames, {encoded[0].width}x{encoded[0].height}, {size_kb:.0f} KiB")


if __name__ == "__main__":
    main()
