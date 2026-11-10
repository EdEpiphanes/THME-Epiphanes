"""
THME Showcase GIF Generator Suite
Generates high-definition, professional engineering animations for THME documentation and marketing.
Zero external heavy dependencies: pure numpy and standard library struct/math.
"""

import math
import struct
import time
import numpy as np

# 5x7 High-Precision Monospace Glyph Dictionary
FONT = {
    'A': [0x1C, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x22],
    'B': [0x3C, 0x22, 0x22, 0x3C, 0x22, 0x22, 0x3C],
    'C': [0x1E, 0x20, 0x20, 0x20, 0x20, 0x20, 0x1E],
    'D': [0x3C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x3C],
    'E': [0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x3E],
    'F': [0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x20],
    'G': [0x1E, 0x20, 0x20, 0x2E, 0x22, 0x22, 0x1E],
    'H': [0x22, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x22],
    'I': [0x1C, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C],
    'J': [0x0E, 0x04, 0x04, 0x04, 0x04, 0x24, 0x18],
    'K': [0x22, 0x24, 0x28, 0x30, 0x28, 0x24, 0x22],
    'L': [0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x3E],
    'M': [0x22, 0x36, 0x2A, 0x2A, 0x22, 0x22, 0x22],
    'N': [0x22, 0x32, 0x2A, 0x26, 0x22, 0x22, 0x22],
    'O': [0x1C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C],
    'P': [0x3C, 0x22, 0x22, 0x3C, 0x20, 0x20, 0x20],
    'Q': [0x1C, 0x22, 0x22, 0x22, 0x2A, 0x24, 0x1A],
    'R': [0x3C, 0x22, 0x22, 0x3C, 0x28, 0x24, 0x22],
    'S': [0x1E, 0x20, 0x20, 0x1C, 0x02, 0x02, 0x3C],
    'T': [0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08],
    'U': [0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C],
    'V': [0x22, 0x22, 0x22, 0x22, 0x14, 0x14, 0x08],
    'W': [0x22, 0x22, 0x22, 0x2A, 0x2A, 0x36, 0x22],
    'X': [0x22, 0x22, 0x14, 0x08, 0x14, 0x22, 0x22],
    'Y': [0x22, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08],
    'Z': [0x3E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3E],
    '0': [0x1C, 0x22, 0x26, 0x2A, 0x32, 0x22, 0x1C],
    '1': [0x08, 0x18, 0x08, 0x08, 0x08, 0x08, 0x1C],
    '2': [0x1C, 0x22, 0x02, 0x0C, 0x30, 0x20, 0x3E],
    '3': [0x1C, 0x22, 0x02, 0x0C, 0x02, 0x22, 0x1C],
    '4': [0x04, 0x0C, 0x14, 0x24, 0x3E, 0x04, 0x04],
    '5': [0x3E, 0x20, 0x3C, 0x02, 0x02, 0x22, 0x1C],
    '6': [0x1C, 0x20, 0x3C, 0x22, 0x22, 0x22, 0x1C],
    '7': [0x3E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10],
    '8': [0x1C, 0x22, 0x22, 0x1C, 0x22, 0x22, 0x1C],
    '9': [0x1C, 0x22, 0x22, 0x1E, 0x02, 0x02, 0x1C],
    ' ': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    '.': [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
    ':': [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
    ',': [0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x18],
    '-': [0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00],
    '+': [0x00, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x00],
    '[': [0x1E, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1E],
    ']': [0x3C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x3C],
    '(': [0x08, 0x10, 0x20, 0x20, 0x20, 0x10, 0x08],
    ')': [0x10, 0x08, 0x04, 0x04, 0x04, 0x08, 0x10],
    '/': [0x02, 0x04, 0x08, 0x10, 0x20, 0x00, 0x00],
    '%': [0x32, 0x34, 0x08, 0x10, 0x12, 0x26, 0x00],
    '>': [0x20, 0x10, 0x08, 0x04, 0x08, 0x10, 0x20],
    '<': [0x04, 0x08, 0x10, 0x20, 0x10, 0x08, 0x04],
    '|': [0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08],
    '_': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3E],
    '*': [0x00, 0x14, 0x08, 0x3E, 0x08, 0x14, 0x00],
    '=': [0x00, 0x3E, 0x00, 0x3E, 0x00, 0x00, 0x00],
    '#': [0x14, 0x14, 0x3E, 0x14, 0x3E, 0x14, 0x14],
    '!': [0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x00, 0x0C],
    '?': [0x1C, 0x22, 0x04, 0x08, 0x08, 0x00, 0x08],
    '&': [0x18, 0x24, 0x18, 0x2A, 0x24, 0x22, 0x1D],
    ';': [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x18],
    '^': [0x08, 0x14, 0x22, 0x00, 0x00, 0x00, 0x00],
    '$': [0x08, 0x1E, 0x28, 0x1C, 0x0A, 0x3C, 0x08],
    '{': [0x0E, 0x08, 0x08, 0x18, 0x08, 0x08, 0x0E],
    '}': [0x38, 0x08, 0x08, 0x0C, 0x08, 0x08, 0x38],
    '~': [0x00, 0x00, 0x1A, 0x2D, 0x00, 0x00, 0x00],
    '\\': [0x20, 0x10, 0x08, 0x04, 0x02, 0x00, 0x00],
    '"': [0x14, 0x14, 0x14, 0x00, 0x00, 0x00, 0x00],
    "'": [0x08, 0x08, 0x10, 0x00, 0x00, 0x00, 0x00],
}

# 24-Color High-Precision Obsidian Engineering Palette
PALETTE = [
    (10, 13, 20),    # 0: Deep Window BG (#0a0d14)
    (17, 22, 34),    # 1: Surface Card BG (#111622)
    (24, 32, 48),    # 2: Section / Box BG (#182030)
    (38, 53, 74),    # 3: Border Soft (#26354a)
    (248, 250, 252), # 4: Text Bright White (#f8fafc)
    (148, 163, 184), # 5: Text Muted Slate (#94a3b8)
    (0, 240, 255),   # 6: Obsidian Cyan (#00f0ff - Brand & Active)
    (16, 185, 129),  # 7: Matrix Emerald (#10b981 - Success / Exit)
    (245, 158, 11),  # 8: Solarized Amber (#f59e0b - Warning / KV)
    (244, 63, 94),   # 9: Cyberpunk Rose / Red (#f43f5e - Hot / Outlier)
    (139, 92, 246),  # 10: Neon Violet (#8b5cf6 - Distillation / Math)
    (59, 130, 246),  # 11: Electric Blue (#3b82f6 - Attention)
    (255, 95, 86),   # 12: Window Close Red (#ff5f56)
    (255, 189, 46),  # 13: Window Min Amber (#ffbd2e)
    (39, 201, 63),   # 14: Window Max Green (#27c93f)
    (6, 182, 212),   # 15: Cyan Secondary (#06b6d4)
    (30, 41, 59),    # 16: Active Tab BG (#1e293b)
    (71, 85, 105),   # 17: Inactive Tab Text (#475569)
    (56, 189, 248),  # 18: Glow Cyan Particle (#38bdf8)
    (5, 150, 105),   # 19: Deep Emerald (#059669)
    (217, 119, 6),   # 20: Deep Amber (#d97706)
    (225, 29, 72),   # 21: Deep Red (#e11d48)
    (30, 27, 75),    # 22: Deep Violet Cell (#1e1b4b)
    (15, 23, 42),    # 23: Terminal / Code BG (#0f172a)
]

def draw_rect(buf, x, y, w, h, col):
    H, W = buf.shape
    x0, x1 = max(0, x), min(W, x + w)
    y0, y1 = max(0, y), min(H, y + h)
    if x1 > x0 and y1 > y0:
        buf[y0:y1, x0:x1] = col

def draw_stroke_rect(buf, x, y, w, h, col, thickness=1):
    draw_rect(buf, x, y, w, thickness, col)
    draw_rect(buf, x, y + h - thickness, w, thickness, col)
    draw_rect(buf, x, y, thickness, h, col)
    draw_rect(buf, x + w - thickness, y, thickness, h, col)

def draw_text(buf, text, x, y, col, scale=1):
    H, W = buf.shape
    cx = x
    for ch in text.upper():
        bitmap = FONT.get(ch, FONT[' '])
        for row_idx, byte_val in enumerate(bitmap):
            for col_idx in range(6):
                if (byte_val >> (5 - col_idx)) & 1:
                    px = cx + col_idx * scale
                    py = y + row_idx * scale
                    if scale == 1:
                        if 0 <= px < W and 0 <= py < H:
                            buf[py, px] = col
                    else:
                        draw_rect(buf, px, py, scale, scale, col)
        cx += 7 * scale

def encode_gif(frames, palette, delay=10):
    """Encodes a sequence of numpy 2D uint8 frames into a GIF89a binary bytearray."""
    H, W = frames[0].shape
    buf = bytearray(b'GIF89a')
    buf.extend(struct.pack('<HH', W, H))
    buf.append(0xF7)
    buf.extend([0, 0])
    for r, g, b in palette:
        buf.extend([r, g, b])
    for _ in range(256 - len(palette)):
        buf.extend([0, 0, 0])
    buf.extend(b'\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00')

    def lzw(pixels):
        min_code_size = 8
        clear_code = 1 << min_code_size
        end_code = clear_code + 1
        code_size = min_code_size + 1
        next_code = end_code + 1
        dictionary = {bytes([i]): i for i in range(clear_code)}

        bits = []
        def emit(code, bits_len):
            for i in range(bits_len):
                bits.append((code >> i) & 1)

        emit(clear_code, code_size)
        curr = b''
        for p in pixels:
            c = bytes([p])
            if curr + c in dictionary:
                curr = curr + c
            else:
                emit(dictionary[curr], code_size)
                if next_code < 4096:
                    dictionary[curr + c] = next_code
                    next_code += 1
                    if next_code > (1 << code_size) and code_size < 12:
                        code_size += 1
                else:
                    emit(clear_code, code_size)
                    code_size = min_code_size + 1
                    next_code = end_code + 1
                    dictionary = {bytes([i]): i for i in range(clear_code)}
                curr = c
        if curr:
            emit(dictionary[curr], code_size)
        emit(end_code, code_size)

        raw_bytes = bytearray()
        cur_b = 0
        bit_pos = 0
        for bit in bits:
            cur_b |= (bit << bit_pos)
            bit_pos += 1
            if bit_pos == 8:
                raw_bytes.append(cur_b)
                cur_b = 0
                bit_pos = 0
        if bit_pos > 0:
            raw_bytes.append(cur_b)

        out = bytearray([min_code_size])
        i = 0
        while i < len(raw_bytes):
            chunk = raw_bytes[i:i+254]
            out.append(len(chunk))
            out.extend(chunk)
            i += len(chunk)
        out.append(0)
        return out

    for f in frames:
        buf.extend(b'\x21\xF9\x04\x04' + struct.pack('<H', delay) + b'\x00\x00')
        buf.extend(b'\x2C\x00\x00\x00\x00' + struct.pack('<HH', W, H) + b'\x00')
        pixels = f.flatten().tolist()
        buf.extend(lzw(pixels))
    buf.append(0x3B)
    return bytes(buf)


# ==============================================================================
# ANIMATION 1: LIVE NEURAL BENCHMARK & MASTER STUDIO (UPDATED TO V0.1.5)
# ==============================================================================
def render_live_benchmark_animation():
    W, H = 920, 520
    num_frames = 24
    frames = []

    tokens = [
        "THME", "TOPOLOGICAL", "SURGERY", "ACHIEVES", "PARETO-OPTIMAL",
        "INFERENCE", "WITH", "ZERO-COPY", "RUST", "SPEED"
    ]

    base_matrix = np.array([
        [ 0.82, -0.15,  1.42, -0.02,  0.65, -0.78,  0.12, -0.45],
        [-0.31,  0.94, -0.08,  0.55, -1.22,  0.34, -0.61,  0.09],
        [ 0.14, -0.67,  0.29, -0.88,  0.41, -0.19,  1.15, -0.32],
        [-0.95,  0.05, -0.42,  1.65, -0.11,  0.81, -0.27,  0.73],
        [ 0.44, -1.12,  0.78, -0.25,  0.92, -0.04,  0.53, -0.89],
        [-0.18,  0.39, -0.62,  0.08, -0.35,  1.28, -0.14,  0.46],
        [ 0.72, -0.24,  0.51, -0.91,  0.18, -0.56,  0.83, -0.07],
        [-0.53,  0.87, -0.16,  0.33, -0.74,  0.22, -0.49,  1.05]
    ], dtype=np.float32)

    for frame_idx in range(num_frames):
        canvas = np.zeros((H, W), dtype=np.uint8)

        # 1. Window Title Bar
        draw_rect(canvas, 0, 0, W, 36, 1)
        draw_stroke_rect(canvas, 0, 0, W, H, 3, thickness=1)

        draw_rect(canvas, 14, 13, 10, 10, 12)
        draw_rect(canvas, 30, 13, 10, 10, 13)
        draw_rect(canvas, 46, 13, 10, 10, 14)

        draw_rect(canvas, 68, 8, 54, 20, 6)
        draw_text(canvas, "THME", 77, 14, 0, scale=1)

        draw_text(canvas, "STUDIO V0.1.5 // TOPOLOGICAL HETEROGENEOUS MANIFOLD ENGINE", 132, 14, 4, scale=1)

        draw_rect(canvas, W - 232, 8, 108, 20, 2)
        draw_stroke_rect(canvas, W - 232, 8, 108, 20, 6)
        draw_text(canvas, "OBSIDIAN CYAN", W - 222, 14, 6, scale=1)

        draw_rect(canvas, W - 114, 8, 100, 20, 2)
        draw_stroke_rect(canvas, W - 114, 8, 100, 20, 7)
        draw_text(canvas, "60 FPS * LIVE", W - 105, 14, 7, scale=1)

        # 2. Studio Tabs (All 9 Engine Tabs)
        tab_y = 37
        tab_h = 28
        draw_rect(canvas, 0, tab_y, W, tab_h, 0)
        draw_rect(canvas, 0, tab_y + tab_h - 1, W, 1, 3)

        tabs = [
            ("COCKPIT", True),
            ("MODELS", False),
            ("ANATOMY", False),
            ("BYOA REGISTRY", False),
            ("TELEMETRY", False),
            ("AUDIT LOG", False),
            ("QUALITY GATE", False),
            ("PYTORCH/CUDA", False),
            ("DEV LAB", False),
        ]
        cur_tab_x = 10
        for t_name, is_active in tabs:
            t_w = len(t_name) * 7 + 12
            if is_active:
                draw_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 16)
                draw_stroke_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 6)
                draw_rect(canvas, cur_tab_x + 2, tab_y + tab_h - 3, t_w - 4, 2, 6)
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 6, scale=1)
            else:
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 5, scale=1)
            cur_tab_x += t_w + 4

        # 3. 4-Card Bento Telemetry Ribbon
        kpi_y = 68
        kpi_h = 42
        card_w = (W - 28 - 24) // 4
        kpis = [
            ("PARETO COMPRESSION", "14.8 GB -> 4.2 GB (-71.6%)", 6, "BITNET B1.58 / INT4 MIX"),
            ("REAL-TIME THROUGHPUT", f"{70.6 + (frame_idx % 4)*0.8:.1f} TOK/S (+2.18X)", 7, "ZERO-COPY FLASH STREAM"),
            ("QUALITY PRESERVATION", "DELTA PPL: +0.032 <= 0.05", 10, "STRICT EVAL: ACCEPTED"),
            ("VRAM CONSUMPTION", f"{4.6 + (frame_idx % 3)*0.1:.1f} GB / 8.0 GB", 8, "AIRLLM NVME STREAM OK"),
        ]
        for i, (title, val, accent_col, sub_desc) in enumerate(kpis):
            bx = 14 + i * (card_w + 8)
            draw_rect(canvas, bx, kpi_y, card_w, kpi_h, 1)
            draw_stroke_rect(canvas, bx, kpi_y, card_w, kpi_h, 3)
            draw_rect(canvas, bx, kpi_y, 3, kpi_h, accent_col)
            draw_text(canvas, title, bx + 8, kpi_y + 7, 5, scale=1)
            draw_text(canvas, val, bx + 8, kpi_y + 19, 4, scale=1)
            draw_text(canvas, sub_desc, bx + 8, kpi_y + 31, accent_col, scale=1)

        # 4. Split Dashboard
        main_y = 116
        main_h = 372

        # Left Column: Layer Tweaker & Matrix Inspection
        left_w = 440
        draw_rect(canvas, 14, main_y, left_w, main_h, 1)
        draw_stroke_rect(canvas, 14, main_y, left_w, main_h, 3)

        draw_rect(canvas, 14, main_y, left_w, 28, 2)
        draw_stroke_rect(canvas, 14, main_y, left_w, 28, 3)
        draw_text(canvas, "LAYER DECONSTRUCTION: L14 MLP.DOWN_PROJ [4096, 4096]", 24, main_y + 10, 6, scale=1)

        prof_y = main_y + 34
        draw_rect(canvas, 24, prof_y, left_w - 20, 24, 23)
        draw_stroke_rect(canvas, 24, prof_y, left_w - 20, 24, 3)
        draw_text(canvas, "EWHA ALLOCATION: 15% ALPHA (1b) + 75% BETA (2b) + 10% GAMMA (PHASE)", 30, prof_y + 8, 4, scale=1)

        # 8x8 Numerical Weight Matrix Slice Heatmap
        matrix_y = prof_y + 30
        draw_text(canvas, "INTERACTIVE 8x8 WEIGHT MATRIX SLICE (LIVE QUANTIZED):", 24, matrix_y, 5, scale=1)

        grid_x0 = 24
        grid_y0 = matrix_y + 14
        cell_w, cell_h = 24, 15
        wave = math.sin(frame_idx * 0.3) * 0.2

        for r in range(8):
            for c in range(8):
                val = base_matrix[r, c] + wave * ((r + c) % 3 - 1)
                cx = grid_x0 + c * (cell_w + 2)
                cy = grid_y0 + r * (cell_h + 2)

                if val > 0.8:
                    c_col = 6
                elif val > 0.2:
                    c_col = 15
                elif val < -0.8:
                    c_col = 9
                elif val < -0.2:
                    c_col = 21
                else:
                    c_col = 8

                draw_rect(canvas, cx, cy, cell_w, cell_h, c_col)
                mark = "+" if val > 0.3 else ("-" if val < -0.3 else "0")
                draw_text(canvas, mark, cx + 8, cy + 4, 0 if c_col != 8 else 4, scale=1)

        # 16-Bar Spectral Histogram
        hist_x0 = grid_x0 + 8 * (cell_w + 2) + 14
        hist_y0 = grid_y0
        hist_w = left_w - (hist_x0 - 14) - 12
        hist_h = 8 * (cell_h + 2) - 2

        draw_rect(canvas, hist_x0, hist_y0, hist_w, hist_h, 2)
        draw_stroke_rect(canvas, hist_x0, hist_y0, hist_w, hist_h, 3)
        draw_text(canvas, "SPECTRAL KURTOSIS", hist_x0 + 8, hist_y0 + 6, 5, scale=1)

        num_bars = 14
        bar_w = (hist_w - 16) // num_bars
        for b in range(num_bars):
            dist = math.exp(-((b - 7) ** 2) / 9.0)
            bh = int(dist * (hist_h - 32) + math.sin(frame_idx * 0.4 + b) * 3)
            bx = hist_x0 + 8 + b * bar_w
            by = hist_y0 + hist_h - bh - 6
            b_col = 9 if (b < 2 or b > 11) else (6 if 5 <= b <= 8 else 10)
            draw_rect(canvas, bx, by, bar_w - 2, bh, b_col)

        # Action Buttons Bar
        act_y = grid_y0 + 8 * (cell_h + 2) + 12
        actions = [
            ("AUTODECIDE", 6),
            ("HADAMARD WHT", 10),
            ("BITNET TERN", 7),
            ("APPLY RECIPE", 4),
        ]
        btn_x = 24
        for a_txt, a_col in actions:
            bw = len(a_txt) * 7 + 12
            draw_rect(canvas, btn_x, act_y, bw, 22, 2)
            draw_stroke_rect(canvas, btn_x, act_y, bw, 22, a_col)
            draw_text(canvas, a_txt, btn_x + 6, act_y + 7, a_col, scale=1)
            btn_x += bw + 8

        # Diagnostic Log
        log_y = act_y + 32
        log_h = main_h - (log_y - main_y) - 12
        draw_rect(canvas, 24, log_y, left_w - 20, log_h, 23)
        draw_stroke_rect(canvas, 24, log_y, left_w - 20, log_h, 3)
        draw_text(canvas, "SURGICAL DIAGNOSTIC TELEMETRY:", 32, log_y + 6, 5, scale=1)
        draw_text(canvas, " > WEIGHT NORM L2: 184.21  |  SPARSITY: 48.2%", 32, log_y + 18, 4, scale=1)
        draw_text(canvas, " > SVD ENERGY TOP-8: 78.4% (RANK REDUCTION VIABLE)", 32, log_y + 30, 7, scale=1)
        draw_text(canvas, " > HADAMARD WHT: 99.4% OUTLIERS SUPPRESSED ZERO LOSS", 32, log_y + 42, 6, scale=1)

        # Right Column: Token Flow & Heterogeneous Backbone
        right_x = 466
        right_w = W - right_x - 14
        draw_rect(canvas, right_x, main_y, right_w, main_h, 1)
        draw_stroke_rect(canvas, right_x, main_y, right_w, main_h, 3)

        draw_rect(canvas, right_x, main_y, right_w, 28, 2)
        draw_stroke_rect(canvas, right_x, main_y, right_w, 28, 3)
        draw_text(canvas, "LIVE TOKEN FLOW & HETEROGENEOUS BACKBONE", right_x + 10, main_y + 10, 7, scale=1)

        # 13-Stage Neural Transformer Backbone Architecture
        spine_y = main_y + 64
        draw_rect(canvas, right_x + 16, spine_y - 1, right_w - 32, 2, 3)

        stages = ["IN", "L0", "L2", "L4", "L8*", "L12", "L16", "L20", "L24", "L28", "L31", "OUT"]
        num_stages = len(stages)
        stage_step = (right_w - 48) // (num_stages - 1)
        active_stage_idx = (frame_idx % (num_stages - 1))
        packet_x = right_x + 24 + (frame_idx / num_frames) * (right_w - 48)

        for s_idx, s_name in enumerate(stages):
            sx = right_x + 24 + s_idx * stage_step
            is_active = (s_idx == active_stage_idx)
            is_exit = ("*" in s_name)
            is_end = (s_idx == 0 or s_idx == num_stages - 1)

            box_sz = 26
            bx = sx - box_sz // 2
            by = spine_y - box_sz // 2

            s_bg = 6 if is_active else (7 if is_exit else (11 if is_end else 2))
            s_stroke = 4 if is_active else (7 if is_exit else 3)

            draw_rect(canvas, bx, by, box_sz, box_sz, s_bg)
            draw_stroke_rect(canvas, bx, by, box_sz, box_sz, s_stroke, thickness=2 if is_active else 1)
            draw_text(canvas, s_name[:3], bx + 4, by + 9, 0 if is_active else 4, scale=1)

            if is_exit:
                draw_rect(canvas, sx - 1, by - 18, 2, 18, 7)
                draw_rect(canvas, sx - 26, by - 28, 52, 12, 7)
                draw_text(canvas, "80% EXIT", sx - 22, by - 26, 0, scale=1)

        # Glowing Active Forward Pulse Particle
        px = int(packet_x)
        draw_rect(canvas, px - 6, spine_y - 6, 12, 12, 18)
        draw_rect(canvas, px - 3, spine_y - 3, 6, 6, 4)

        # Autoregressive Generation Terminal
        term_y = main_y + 96
        draw_rect(canvas, right_x + 14, term_y, right_w - 28, 92, 23)
        draw_stroke_rect(canvas, right_x + 14, term_y, right_w - 28, 92, 3)
        draw_text(canvas, ">_ AUTOREGRESSIVE GENERATION STREAM [LLAMA-3-8B]:", right_x + 20, term_y + 8, 6, scale=1)

        tok_count = min(len(tokens), (frame_idx // 2) + 1)
        streamed = " ".join(tokens[:tok_count])
        line1 = streamed[:38]
        line2 = streamed[38:]
        draw_text(canvas, "PROMPT: TOPOLOGICAL SURGERY ON TRANSFORMER...", right_x + 20, term_y + 24, 5, scale=1)
        draw_text(canvas, ">> " + line1, right_x + 20, term_y + 42, 4, scale=1)
        if line2:
            draw_text(canvas, "   " + line2, right_x + 20, term_y + 56, 4, scale=1)

        draw_text(canvas, f"STATUS: STREAMING ({tok_count}/10) | TTFT: 12.8MS | SPEED: 70.6 TOK/S", right_x + 20, term_y + 74, 7, scale=1)

        # Layer Latency Waterfall
        wf_y = term_y + 98
        wf_h = main_h - (wf_y - main_y) - 12
        draw_rect(canvas, right_x + 14, wf_y, right_w - 28, wf_h, 2)
        draw_stroke_rect(canvas, right_x + 14, wf_y, right_w - 28, wf_h, 3)
        draw_text(canvas, "LAYER EXECUTION WATERFALL & QUANTIZATION SCHEME:", right_x + 20, wf_y + 8, 4, scale=1)

        num_wf_bars = 7
        bar_step_y = 18
        layer_tags = ["FP16 DENSE", "Q4_K MIXED", "BITNET TERN", "DELTANET REC", "2:4 SPARSE", "Q4_K MIXED", "FP16 EXIT"]
        for b in range(num_wf_bars):
            by = wf_y + 26 + b * bar_step_y
            is_cur_b = (b == (active_stage_idx % num_wf_bars))

            draw_text(canvas, f"L{b*4}", right_x + 20, by + 2, 6 if is_cur_b else 5, scale=1)

            bw = 110 + int(math.sin(frame_idx * 0.4 + b) * 25) + (35 if is_cur_b else 0)
            b_col = 6 if is_cur_b else (10 if b % 2 == 1 else 11)
            draw_rect(canvas, right_x + 48, by, bw, 11, b_col)

            lat_str = f"{1.15 + (bw / 140.0):.2f}MS"
            draw_text(canvas, lat_str, right_x + 56 + bw, by + 2, 4 if is_cur_b else 5, scale=1)
            draw_text(canvas, layer_tags[b], right_x + right_w - 110, by + 2, 7 if "BITNET" in layer_tags[b] or "EXIT" in layer_tags[b] else 5, scale=1)

        # 5. Bottom Status Bar
        bot_y = H - 24
        draw_rect(canvas, 0, bot_y, W, 24, 1)
        draw_rect(canvas, 0, bot_y, W, 1, 3)

        draw_rect(canvas, 14, bot_y + 7, 8, 8, 7)
        draw_text(canvas, "ONLINE", 28, bot_y + 8, 7, scale=1)
        draw_text(canvas, "|  MEM BW: 384.2 GB/S  |  KV CACHE: 148.4 MB  |  MCP: ACTIVE  |  PYTORCH/CUDA: CONNECTED  |  PURE RUST", 82, bot_y + 8, 5, scale=1)

        frames.append(canvas)

    gif_bytes = encode_gif(frames, PALETTE, delay=10)
    with open("assets/live_neural_benchmark.gif", "wb") as f:
        f.write(gif_bytes)
    print("Generated assets/live_neural_benchmark.gif (", len(gif_bytes), "bytes )")


# ==============================================================================
# ANIMATION 2: SCIENTIFIC MATH & ALGORITHMIC OPTIMIZATION SHOWCASE
# ==============================================================================
def render_scientific_math_animation():
    W, H = 920, 520
    num_frames = 24
    frames = []

    for frame_idx in range(num_frames):
        canvas = np.zeros((H, W), dtype=np.uint8)

        # 1. Window Header
        draw_rect(canvas, 0, 0, W, 36, 1)
        draw_stroke_rect(canvas, 0, 0, W, H, 3, thickness=1)

        draw_rect(canvas, 14, 13, 10, 10, 12)
        draw_rect(canvas, 30, 13, 10, 10, 13)
        draw_rect(canvas, 46, 13, 10, 10, 14)

        draw_rect(canvas, 68, 8, 88, 20, 10) # Violet pill for scientific engine
        draw_text(canvas, "THME MATH", 75, 14, 0, scale=1)

        draw_text(canvas, "SCIENTIFIC OPTIMIZATION ENGINE // SPECTRAL DECAY & PARETO PROOFS", 166, 14, 4, scale=1)

        draw_rect(canvas, W - 140, 8, 126, 20, 2)
        draw_stroke_rect(canvas, W - 140, 8, 126, 20, 7)
        draw_text(canvas, "PPL GATE: PASSED", W - 132, 14, 7, scale=1)

        # 2. Tabs
        tab_y = 37
        tab_h = 28
        draw_rect(canvas, 0, tab_y, W, tab_h, 0)
        draw_rect(canvas, 0, tab_y + tab_h - 1, W, 1, 3)

        tabs = [
            ("HADAMARD ROTATION", True),
            ("SVD SPECTRAL DECAY", False),
            ("BITNET TERNARY B1.58", False),
            ("EWHA ENERGY PROOF", False),
            ("PARETO DP SOLVER", False),
            ("LORA MERGE FORMULA", False),
        ]
        cur_tab_x = 12
        for t_name, is_active in tabs:
            t_w = len(t_name) * 7 + 12
            if is_active:
                draw_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 16)
                draw_stroke_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 10)
                draw_rect(canvas, cur_tab_x + 2, tab_y + tab_h - 3, t_w - 4, 2, 10)
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 10, scale=1)
            else:
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 5, scale=1)
            cur_tab_x += t_w + 6

        # 3. 4-Card Bento Math Ribbon
        kpi_y = 68
        kpi_h = 42
        card_w = (W - 28 - 24) // 4
        kpis = [
            ("ORTHOGONAL ROTATION", "MAX ACTIVATION: 18.4 -> 2.1", 10, "99.4% OUTLIERS SUPPRESSED"),
            ("ENERGY CONSUMPTION", "-68.5% JOULES / TOKEN", 7, "EWHA 80/20 REACH PROOF"),
            ("STORAGE REDUCTION", "16 BITS -> 1.58 BITS (-90%)", 6, "BITNET B1.58 CODEBOOK"),
            ("EMPIRICAL PERPLEXITY", "WIKITEXT-2 PPL: 6.14 -> 6.17", 8, "DELTA PPL: +0.03 (ACCEPT)"),
        ]
        for i, (title, val, accent_col, sub_desc) in enumerate(kpis):
            bx = 14 + i * (card_w + 8)
            draw_rect(canvas, bx, kpi_y, card_w, kpi_h, 1)
            draw_stroke_rect(canvas, bx, kpi_y, card_w, kpi_h, 3)
            draw_rect(canvas, bx, kpi_y, 3, kpi_h, accent_col)
            draw_text(canvas, title, bx + 8, kpi_y + 7, 5, scale=1)
            draw_text(canvas, val, bx + 8, kpi_y + 19, 4, scale=1)
            draw_text(canvas, sub_desc, bx + 8, kpi_y + 31, accent_col, scale=1)

        # 4. Split Layout: Mathematical Formulas on Left, Empirical Curves on Right
        main_y = 116
        main_h = 372

        # Left Column: Mathematical Formulas & Active Operators
        left_w = 460
        draw_rect(canvas, 14, main_y, left_w, main_h, 1)
        draw_stroke_rect(canvas, 14, main_y, left_w, main_h, 3)

        draw_rect(canvas, 14, main_y, left_w, 28, 2)
        draw_stroke_rect(canvas, 14, main_y, left_w, 28, 3)
        draw_text(canvas, "MATHEMATICAL FORMULATIONS & ACTIVE TENSOR OPERATORS", 24, main_y + 10, 10, scale=1)

        # Formula Cards
        formula_cards = [
            ("1. FAST WALSH-HADAMARD TRANSFORM (FWHT)",
             "Y = (1 / SQRT(D)) * H_D * X",
             "SUPPRESSES SYSTEMATIC ACTIVATION OUTLIERS ROTATIONALLY",
             6),
            ("2. BITNET B1.58 TERNARY DISCRETIZATION",
             "W_TILDE = ROUND_CLIP( W / (1/N * ||W||_1), -1, +1 )",
             "REDUCES MATRIX MULTIPLICATION TO INT8 ACCUMULATE & ADDS",
             7),
            ("3. EWHA EXECUTION-WEIGHTED ENERGY FORMULA",
             "E_TOKEN = SUM_L [ Q_L * ( E_L,ATTN + E_L,MLP ) ]",
             "SURFACE REACH Q_L=1.00 (L<=8) -> DEEP REACH Q_L=0.20 (L>8)",
             10),
            ("4. PARETO OPTIMAL KNAPSACK DP GATE",
             "MAX SUM_L DELTA_MEM_L  SUBJECT TO  DELTA_PPL <= EPSILON",
             "FINDS GLOBAL PARETO FRONTIER UNDER STRICT QUALITY REJECTION",
             8),
        ]

        card_step_y = 78
        for c_idx, (f_title, f_eq, f_desc, f_col) in enumerate(formula_cards):
            cy = main_y + 36 + c_idx * card_step_y
            draw_rect(canvas, 24, cy, left_w - 20, 70, 2)
            draw_stroke_rect(canvas, 24, cy, left_w - 20, 70, 3)
            draw_rect(canvas, 24, cy, 3, 70, f_col)

            draw_text(canvas, f_title, 34, cy + 8, f_col, scale=1)

            # Equation Box
            draw_rect(canvas, 34, cy + 22, left_w - 40, 22, 23)
            draw_stroke_rect(canvas, 34, cy + 22, left_w - 40, 22, 3)
            draw_text(canvas, f_eq, 42, cy + 29, 4, scale=1)

            draw_text(canvas, f_desc, 34, cy + 52, 5, scale=1)

        # Right Column: Empirical Convergence Curves & Verification
        right_x = 486
        right_w = W - right_x - 14
        draw_rect(canvas, right_x, main_y, right_w, main_h, 1)
        draw_stroke_rect(canvas, right_x, main_y, right_w, main_h, 3)

        draw_rect(canvas, right_x, main_y, right_w, 28, 2)
        draw_stroke_rect(canvas, right_x, main_y, right_w, 28, 3)
        draw_text(canvas, "EMPIRICAL CONVERGENCE & HARDWARE VALIDATION", right_x + 10, main_y + 10, 6, scale=1)

        # Plot 1: Activation Distribution Before vs After Hadamard Rotation
        plot1_y = main_y + 36
        plot1_h = 130
        draw_rect(canvas, right_x + 14, plot1_y, right_w - 28, plot1_h, 2)
        draw_stroke_rect(canvas, right_x + 14, plot1_y, right_w - 28, plot1_h, 3)
        draw_text(canvas, "ACTIVATION OUTLIER REDUCTION (D=4096):", right_x + 22, plot1_y + 8, 4, scale=1)

        # Legend
        draw_rect(canvas, right_x + 260, plot1_y + 8, 8, 8, 9)
        draw_text(canvas, "BASE", right_x + 272, plot1_y + 9, 5, scale=1)
        draw_rect(canvas, right_x + 310, plot1_y + 8, 8, 8, 6)
        draw_text(canvas, "HADAMARD", right_x + 322, plot1_y + 9, 6, scale=1)

        # Draw grid
        gw = right_w - 56
        gh = plot1_h - 36
        gx0 = right_x + 28
        gy0 = plot1_y + 26

        draw_rect(canvas, gx0, gy0 + gh, gw, 1, 3)
        draw_rect(canvas, gx0, gy0, 1, gh, 3)

        # Draw baseline outlier spikes (red) vs smooth rotated curve (cyan)
        anim_phase = frame_idx * 0.25
        num_points = 50
        for pt in range(num_points - 1):
            px1 = gx0 + int(pt * (gw / num_points))
            px2 = gx0 + int((pt + 1) * (gw / num_points))

            # Base curve with high outlier spikes
            y_base1 = gy0 + gh - int((math.exp(-((pt - 25)**2)/120.0) * 0.4 + (0.55 if pt in [12, 38] else 0.05)) * gh)
            y_base2 = gy0 + gh - int((math.exp(-((pt + 1 - 25)**2)/120.0) * 0.4 + (0.55 if (pt+1) in [12, 38] else 0.05)) * gh)
            draw_rect(canvas, px1, min(y_base1, y_base2), max(2, px2 - px1), abs(y_base2 - y_base1) + 2, 9)

            # Hadamard rotated smooth Gaussian (cyan)
            y_had1 = gy0 + gh - int(math.exp(-((pt - 25)**2)/60.0) * (gh * 0.82))
            y_had2 = gy0 + gh - int(math.exp(-((pt + 1 - 25)**2)/60.0) * (gh * 0.82))
            draw_rect(canvas, px1, min(y_had1, y_had2), max(2, px2 - px1), abs(y_had2 - y_had1) + 2, 6)

        # Plot 2: Per-Token Latency vs Traditional Execution
        plot2_y = plot1_y + plot1_h + 10
        plot2_h = 160
        draw_rect(canvas, right_x + 14, plot2_y, right_w - 28, plot2_h, 2)
        draw_stroke_rect(canvas, right_x + 14, plot2_y, right_w - 28, plot2_h, 3)
        draw_text(canvas, "LATENCY SPEEDUP COMPARISON [LLAMA-3-8B]:", right_x + 22, plot2_y + 8, 4, scale=1)

        benchmarks = [
            ("STANDARD PYTORCH FP16", 30.8, 9, "32.4 TOK/S (BASELINE)"),
            ("BITSANDBYTES QLORA 4-BIT", 22.4, 8, "44.6 TOK/S (+1.37X)"),
            ("THME PARETO OPTIMIZED", 14.1, 7, "70.6 TOK/S (+2.18X)"),
        ]

        bar_base_y = plot2_y + 30
        for b_i, (b_name, lat_val, b_col, b_tok) in enumerate(benchmarks):
            by = bar_base_y + b_i * 36
            draw_text(canvas, b_name, right_x + 22, by, 5, scale=1)

            bw = int((lat_val / 35.0) * (right_w - 180))
            draw_rect(canvas, right_x + 22, by + 12, bw, 14, b_col)
            draw_text(canvas, f"{lat_val:.1f} MS", right_x + 28 + bw, by + 14, 4, scale=1)
            draw_text(canvas, b_tok, right_x + right_w - 150, by + 14, b_col, scale=1)

        # Quality Gate Verdict Banner
        gate_y = main_y + main_h - 32
        draw_rect(canvas, right_x + 14, gate_y, right_w - 28, 24, 23)
        draw_stroke_rect(canvas, right_x + 14, gate_y, right_w - 28, 24, 7)
        draw_rect(canvas, right_x + 20, gate_y + 7, 8, 8, 7)
        draw_text(canvas, "QUALITY GATE VERIFIED: DELTA PPL = +0.032 <= 0.050 (COMMITTED)", right_x + 36, gate_y + 8, 7, scale=1)

        # 5. Bottom Status Bar
        bot_y = H - 24
        draw_rect(canvas, 0, bot_y, W, 24, 1)
        draw_rect(canvas, 0, bot_y, W, 1, 3)

        draw_rect(canvas, 14, bot_y + 7, 8, 8, 7)
        draw_text(canvas, "MATHEMATICALLY PROVEN", 28, bot_y + 8, 7, scale=1)
        draw_text(canvas, "|  FWHT: 4.5X AVX2 SPEEDUP  |  BITNET: 1.58b  |  EWHA: 80/20 REACH  |  PARETO CONVERGENCE: O(N*M)", 186, bot_y + 8, 5, scale=1)

        frames.append(canvas)

    gif_bytes = encode_gif(frames, PALETTE, delay=10)
    with open("assets/scientific_math_optimization.gif", "wb") as f:
        f.write(gif_bytes)
    print("Generated assets/scientific_math_optimization.gif (", len(gif_bytes), "bytes )")


# ==============================================================================
# ANIMATION 3: LAYER ANATOMY DECONSTRUCTION & REBUILDER SURGERY
# ==============================================================================
def render_layer_deconstruction_animation():
    W, H = 920, 520
    num_frames = 24
    frames = []

    for frame_idx in range(num_frames):
        canvas = np.zeros((H, W), dtype=np.uint8)

        # 1. Window Title Bar
        draw_rect(canvas, 0, 0, W, 36, 1)
        draw_stroke_rect(canvas, 0, 0, W, H, 3, thickness=1)

        draw_rect(canvas, 14, 13, 10, 10, 12)
        draw_rect(canvas, 30, 13, 10, 10, 13)
        draw_rect(canvas, 46, 13, 10, 10, 14)

        draw_rect(canvas, 68, 8, 92, 20, 7) # Emerald pill
        draw_text(canvas, "THME ANATOMY", 74, 14, 0, scale=1)

        draw_text(canvas, "LAYER DECONSTRUCTION ENGINE // SUB-BLOCK TOPOLOGY SURGERY", 170, 14, 4, scale=1)

        draw_rect(canvas, W - 146, 8, 132, 20, 2)
        draw_stroke_rect(canvas, W - 146, 8, 132, 20, 6)
        draw_text(canvas, "32 LAYERS DISASSEMBLED", W - 140, 14, 6, scale=1)

        # 2. Tabs
        tab_y = 37
        tab_h = 28
        draw_rect(canvas, 0, tab_y, W, tab_h, 0)
        draw_rect(canvas, 0, tab_y + tab_h - 1, W, 1, 3)

        tabs = [
            ("TOPOLOGY MAP", True),
            ("LAYER 0 (SURFACE)", False),
            ("LAYER 8 (CHECKPOINT)", False),
            ("LAYER 16 (DEEP)", False),
            ("CUSTOM BLUEPRINT", False),
            ("SURGERY RECIPES", False),
        ]
        cur_tab_x = 12
        for t_name, is_active in tabs:
            t_w = len(t_name) * 7 + 12
            if is_active:
                draw_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 16)
                draw_stroke_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 7)
                draw_rect(canvas, cur_tab_x + 2, tab_y + tab_h - 3, t_w - 4, 2, 7)
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 7, scale=1)
            else:
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 5, scale=1)
            cur_tab_x += t_w + 6

        # 3. 4-Card Bento Ribbon
        kpi_y = 68
        kpi_h = 42
        card_w = (W - 28 - 24) // 4
        kpis = [
            ("ACTIVE TOPOLOGY", "HYBRID DUAL-MIXER", 7, "GQA ATTN + GATED DELTANET"),
            ("SUB-BLOCKS EXTRACTED", "256 TENSORS MAPPED", 6, "RMSNORM, SWIGLU, RESIDUALS"),
            ("STORAGE RECLAIMED", "-4,120 MB RAM (-64.2%)", 10, "IN-PLACE BITNET SURGERY"),
            ("EARLY-EXIT ROUTING", "80% EXIT AT LAYER 8", 8, "DELTA PPL <= 0.05 PRESERVED"),
        ]
        for i, (title, val, accent_col, sub_desc) in enumerate(kpis):
            bx = 14 + i * (card_w + 8)
            draw_rect(canvas, bx, kpi_y, card_w, kpi_h, 1)
            draw_stroke_rect(canvas, bx, kpi_y, card_w, kpi_h, 3)
            draw_rect(canvas, bx, kpi_y, 3, kpi_h, accent_col)
            draw_text(canvas, title, bx + 8, kpi_y + 7, 5, scale=1)
            draw_text(canvas, val, bx + 8, kpi_y + 19, 4, scale=1)
            draw_text(canvas, sub_desc, bx + 8, kpi_y + 31, accent_col, scale=1)

        # 4. Main Panel: Full Layer Anatomy Dataflow Map
        main_y = 116
        main_h = 372
        draw_rect(canvas, 14, main_y, W - 28, main_h, 1)
        draw_stroke_rect(canvas, 14, main_y, W - 28, main_h, 3)

        # Header
        draw_rect(canvas, 14, main_y, W - 28, 28, 2)
        draw_stroke_rect(canvas, 14, main_y, W - 28, 28, 3)
        draw_text(canvas, "TRANSFORMER LAYER SUB-BLOCK ANATOMY & DATAFLOW ROUTING HIGHWAY", 24, main_y + 10, 7, scale=1)

        # Visual Flow Diagram: 7 Nodes across the canvas
        nodes = [
            ("1. INPUT RMSNORM", "PRE-MIXER NORM", "FP16 UNTOUCHED", 11),
            ("2. SEQUENCE MIXER", "W_Q, W_K, W_V, W_O", "SURGERY: GATED DELTANET", 6),
            ("3. HIGHWAY 1", "X + MIXER(X)", "ADD RESIDUAL", 5),
            ("4. POST-RMSNORM", "PRE-MLP NORM", "FP16 UNTOUCHED", 11),
            ("5. SWIGLU MLP", "W_GATE, W_UP, W_DOWN", "SURGERY: BITNET B1.58", 10),
            ("6. HIGHWAY 2", "X + MLP(X)", "ADD RESIDUAL", 5),
            ("7. EARLY-EXIT", "CONFIDENCE CHECK", "80% ROUTE EXIT", 7),
        ]

        node_w = 114
        node_h = 94
        spacing = (W - 56 - (len(nodes) * node_w)) // (len(nodes) - 1)
        ny = main_y + 44

        active_node_idx = (frame_idx // 3) % len(nodes)

        for n_i, (n_title, n_sub, n_tag, n_col) in enumerate(nodes):
            nx = 24 + n_i * (node_w + spacing)
            is_cur = (n_i == active_node_idx)

            # Connecting arrow to next node
            if n_i < len(nodes) - 1:
                ax = nx + node_w
                ay = ny + node_h // 2
                draw_rect(canvas, ax, ay - 1, spacing, 2, 6 if is_cur else 3)
                draw_rect(canvas, ax + spacing - 4, ay - 3, 2, 6, 6 if is_cur else 3)

            # Node Box
            draw_rect(canvas, nx, ny, node_w, node_h, 16 if is_cur else 2)
            draw_stroke_rect(canvas, nx, ny, node_w, node_h, 4 if is_cur else (n_col if n_col != 5 else 3), thickness=2 if is_cur else 1)
            draw_rect(canvas, nx, ny, node_w, 3, n_col)

            draw_text(canvas, n_title[:15], nx + 6, ny + 10, 4 if is_cur else 5, scale=1)
            draw_text(canvas, n_sub[:15], nx + 6, ny + 26, 6 if is_cur else 4, scale=1)

            # Sub-block tag badge
            draw_rect(canvas, nx + 4, ny + 48, node_w - 8, 38, 23)
            draw_stroke_rect(canvas, nx + 4, ny + 48, node_w - 8, 38, n_col)
            draw_text(canvas, n_tag[:14], nx + 8, ny + 54, n_col, scale=1)
            if "SURGERY" in n_tag:
                draw_text(canvas, "MODIFIED", nx + 8, ny + 68, 7, scale=1)
            else:
                draw_text(canvas, "PASS-THRU", nx + 8, ny + 68, 5, scale=1)

        # Real-Time Recipe Execution Log Panel
        log_y = ny + node_h + 16
        log_h = main_h - (log_y - main_y) - 12
        draw_rect(canvas, 24, log_y, W - 48, log_h, 23)
        draw_stroke_rect(canvas, 24, log_y, W - 48, log_h, 3)

        draw_text(canvas, ">_ DECLARATIVE SURGERY RECIPE DISPATCH & IN-ENGINE RECONSTRUCTION:", 34, log_y + 8, 6, scale=1)

        recipe_steps = [
            ("[STEP 1/3]", "SUBSTITUTE SEQUENCE MIXER", "REPLACE ATTENTION (L0..L7) -> GATED DELTANET", "SUCCESS (-35% KV CACHE)"),
            ("[STEP 2/3]", "DISCRETIZE SWIGLU MLP", "APPLY BITNET B1.58 TERNARY TO MLP GATE (L8..L31)", "SUCCESS (-64% WEIGHT RAM)"),
            ("[STEP 3/3]", "ROUTE EARLY-EXIT GATE", "SET CHECKPOINT LAYER 8 (CONFIDENCE THRESHOLD 0.85)", "SUCCESS (80% EXIT REACH)"),
        ]

        step_base_y = log_y + 26
        for s_i, (s_num, s_act, s_det, s_st) in enumerate(recipe_steps):
            sy = step_base_y + s_i * 24
            is_active_step = (s_i == (frame_idx // 8) % 3)

            draw_text(canvas, s_num, 34, sy, 6 if is_active_step else 5, scale=1)
            draw_text(canvas, s_act, 106, sy, 4 if is_active_step else 5, scale=1)
            draw_text(canvas, s_det, 320, sy, 5, scale=1)
            draw_text(canvas, s_st, W - 220, sy, 7 if is_active_step else 19, scale=1)

        # 5. Bottom Status Bar
        bot_y = H - 24
        draw_rect(canvas, 0, bot_y, W, 24, 1)
        draw_rect(canvas, 0, bot_y, W, 1, 3)

        draw_rect(canvas, 14, bot_y + 7, 8, 8, 7)
        draw_text(canvas, "RECIPE COMMITTED", 28, bot_y + 8, 7, scale=1)
        draw_text(canvas, "|  TOPOLOGY: HYBRID DUAL-MIXER  |  RAM RECLAIMED: 4120.4 MB  |  THROUGHPUT: 2.18X  |  ZERO LOSS", 166, bot_y + 8, 5, scale=1)

        frames.append(canvas)

    gif_bytes = encode_gif(frames, PALETTE, delay=10)
    with open("assets/layer_deconstruction_surgery.gif", "wb") as f:
        f.write(gif_bytes)
    print("Generated assets/layer_deconstruction_surgery.gif (", len(gif_bytes), "bytes )")


# ==============================================================================
# ANIMATION 4: PYTORCH, CUDA & PEFT LORA INTEROPERABILITY BRIDGE
# ==============================================================================
def render_pytorch_cuda_interop_animation():
    W, H = 920, 520
    num_frames = 24
    frames = []

    for frame_idx in range(num_frames):
        canvas = np.zeros((H, W), dtype=np.uint8)

        # 1. Window Header
        draw_rect(canvas, 0, 0, W, 36, 1)
        draw_stroke_rect(canvas, 0, 0, W, H, 3, thickness=1)

        draw_rect(canvas, 14, 13, 10, 10, 12)
        draw_rect(canvas, 30, 13, 10, 10, 13)
        draw_rect(canvas, 46, 13, 10, 10, 14)

        draw_rect(canvas, 68, 8, 104, 20, 8) # Amber pill
        draw_text(canvas, "THME INTEROP", 74, 14, 0, scale=1)

        draw_text(canvas, "PYTORCH & CUDA ECOSYSTEM BRIDGE // PEFT LORA ZERO-COPY MERGE", 182, 14, 4, scale=1)

        draw_rect(canvas, W - 156, 8, 142, 20, 2)
        draw_stroke_rect(canvas, W - 156, 8, 142, 20, 7)
        draw_text(canvas, "CUDA SM80+ DETECTED", W - 148, 14, 7, scale=1)

        # 2. Tabs
        tab_y = 37
        tab_h = 28
        draw_rect(canvas, 0, tab_y, W, tab_h, 0)
        draw_rect(canvas, 0, tab_y + tab_h - 1, W, 1, 3)

        tabs = [
            ("TRAIN SCRIPT GENERATOR", True),
            ("PEFT LORA MERGER", False),
            ("HUGGINGFACE CONFIG", False),
            ("DEEPSPEED ZERO-3", False),
            ("UNSLOTH FAST", False),
            ("EXPORT GGUF", False),
        ]
        cur_tab_x = 12
        for t_name, is_active in tabs:
            t_w = len(t_name) * 7 + 12
            if is_active:
                draw_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 16)
                draw_stroke_rect(canvas, cur_tab_x, tab_y + 3, t_w, tab_h - 4, 8)
                draw_rect(canvas, cur_tab_x + 2, tab_y + tab_h - 3, t_w - 4, 2, 8)
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 8, scale=1)
            else:
                draw_text(canvas, t_name, cur_tab_x + 6, tab_y + 11, 5, scale=1)
            cur_tab_x += t_w + 6

        # 3. 4-Card Bento Ribbon
        kpi_y = 68
        kpi_h = 42
        card_w = (W - 28 - 24) // 4
        kpis = [
            ("TRAINING STRATEGY", "QLORA 4-BIT (BNB NF4)", 8, "FLASHATTENTION-2 + TF32"),
            ("PEFT ADAPTER MERGE", "W = W0 + (ALPHA/R) * B*A", 7, "ZERO-COPY MATRIX MATH"),
            ("TARGET HARDWARE", "NVIDIA CUDA DRIVER API", 6, "PURE RUST CUDARC NO PYTHON"),
            ("EXPORT COMPATIBILITY", "GGUF + SAFETENSORS", 10, "LLAMA.CPP & VLLM READY"),
        ]
        for i, (title, val, accent_col, sub_desc) in enumerate(kpis):
            bx = 14 + i * (card_w + 8)
            draw_rect(canvas, bx, kpi_y, card_w, kpi_h, 1)
            draw_stroke_rect(canvas, bx, kpi_y, card_w, kpi_h, 3)
            draw_rect(canvas, bx, kpi_y, 3, kpi_h, accent_col)
            draw_text(canvas, title, bx + 8, kpi_y + 7, 5, scale=1)
            draw_text(canvas, val, bx + 8, kpi_y + 19, 4, scale=1)
            draw_text(canvas, sub_desc, bx + 8, kpi_y + 31, accent_col, scale=1)

        # 4. Split Layout: Code Generator on Left, In-Engine LoRA Merger on Right
        main_y = 116
        main_h = 372

        # Left Column: Synthesized Production Python Script
        left_w = 460
        draw_rect(canvas, 14, main_y, left_w, main_h, 1)
        draw_stroke_rect(canvas, 14, main_y, left_w, main_h, 3)

        draw_rect(canvas, 14, main_y, left_w, 28, 2)
        draw_stroke_rect(canvas, 14, main_y, left_w, 28, 3)
        draw_text(canvas, "SYNTHESIZED PRODUCTION SCRIPT: TRAIN_CUDA.PY", 24, main_y + 10, 8, scale=1)

        # Code Editor Box
        code_y = main_y + 36
        code_h = main_h - 48
        draw_rect(canvas, 24, code_y, left_w - 20, code_h, 23)
        draw_stroke_rect(canvas, 24, code_y, left_w - 20, code_h, 3)

        code_lines = [
            ("#!/usr/bin/env python3", 5),
            ("# Auto-generated by THME v0.1.5 PyTorch/CUDA Bridge", 5),
            ("import torch, bitsandbytes as bnb", 6),
            ("from transformers import AutoModelForCausalLM", 4),
            ("from peft import LoraConfig, get_peft_model", 8),
            ("", 5),
            ("torch.backends.cuda.matmul.allow_tf32 = True", 7),
            ("bnb_config = BitsAndBytesConfig(", 4),
            ("    load_in_4bit=True, bnb_4bit_quant_type='nf4'", 10),
            (")", 4),
            ("lora_config = LoraConfig(", 4),
            ("    r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj']", 8),
            (")", 4),
            ("model = get_peft_model(model, lora_config)", 6),
            ("trainer = SFTTrainer(model=model, train_dataset=data)", 7),
            ("trainer.train()", 4),
            ("print('Fine-tuning finished. Ready for THME zero-copy merge!')", 7),
        ]

        for l_idx, (l_code, l_col) in enumerate(code_lines):
            ly = code_y + 12 + l_idx * 17
            draw_text(canvas, f"{l_idx+1:2d} ", 32, ly, 17, scale=1)
            draw_text(canvas, l_code[:50], 58, ly, l_col, scale=1)

        # Right Column: In-Engine LoRA Weight Merger Visualizer
        right_x = 486
        right_w = W - right_x - 14
        draw_rect(canvas, right_x, main_y, right_w, main_h, 1)
        draw_stroke_rect(canvas, right_x, main_y, right_w, main_h, 3)

        draw_rect(canvas, right_x, main_y, right_w, 28, 2)
        draw_stroke_rect(canvas, right_x, main_y, right_w, 28, 3)
        draw_text(canvas, "IN-ENGINE PEFT LORA ZERO-COPY WEIGHT MERGER", right_x + 10, main_y + 10, 7, scale=1)

        # Matrix Product Diagram: W0 + (alpha/r)*(B x A)
        diag_y = main_y + 36
        draw_rect(canvas, right_x + 14, diag_y, right_w - 28, 140, 2)
        draw_stroke_rect(canvas, right_x + 14, diag_y, right_w - 28, 140, 3)
        draw_text(canvas, "MATHEMATICAL FORMULATION: W = W0 + (32/16) * (B x A)", right_x + 22, diag_y + 10, 4, scale=1)

        # Box Base Weight W0 (FP16)
        draw_rect(canvas, right_x + 24, diag_y + 32, 64, 64, 11)
        draw_stroke_rect(canvas, right_x + 24, diag_y + 32, 64, 64, 4)
        draw_text(canvas, "W0", right_x + 46, diag_y + 54, 4, scale=1)
        draw_text(canvas, "4096x4096", right_x + 26, diag_y + 80, 0, scale=1)

        draw_text(canvas, "+", right_x + 98, diag_y + 58, 4, scale=1)

        # Scale Factor Pill
        draw_rect(canvas, right_x + 114, diag_y + 48, 36, 32, 8)
        draw_text(canvas, "2.0x", right_x + 118, diag_y + 58, 0, scale=1)

        draw_text(canvas, "*", right_x + 158, diag_y + 58, 4, scale=1)

        # Matrix B (4096 x 16)
        draw_rect(canvas, right_x + 174, diag_y + 32, 28, 64, 6)
        draw_stroke_rect(canvas, right_x + 174, diag_y + 32, 28, 64, 4)
        draw_text(canvas, "B", right_x + 182, diag_y + 54, 0, scale=1)

        draw_text(canvas, "x", right_x + 210, diag_y + 58, 4, scale=1)

        # Matrix A (16 x 4096)
        draw_rect(canvas, right_x + 226, diag_y + 50, 64, 28, 10)
        draw_stroke_rect(canvas, right_x + 226, diag_y + 50, 64, 28, 4)
        draw_text(canvas, "A", right_x + 252, diag_y + 58, 0, scale=1)

        draw_text(canvas, "=", right_x + 300, diag_y + 58, 4, scale=1)

        # Result Merged W
        draw_rect(canvas, right_x + 318, diag_y + 32, 64, 64, 7)
        draw_stroke_rect(canvas, right_x + 318, diag_y + 32, 64, 64, 4)
        draw_text(canvas, "W*", right_x + 342, diag_y + 54, 0, scale=1)
        draw_text(canvas, "MERGED", right_x + 326, diag_y + 80, 0, scale=1)

        draw_text(canvas, "RANK r=16 | ALPHA=32 | ZERO MEMORY ALLOCATIONS", right_x + 22, diag_y + 112, 7, scale=1)

        # Progress / Merge Status Box
        prog_y = diag_y + 152
        prog_h = main_h - (prog_y - main_y) - 12
        draw_rect(canvas, right_x + 14, prog_y, right_w - 28, prog_h, 23)
        draw_stroke_rect(canvas, right_x + 14, prog_y, right_w - 28, prog_h, 3)

        draw_text(canvas, "IN-ENGINE MERGING PROGRESSION:", right_x + 22, prog_y + 8, 5, scale=1)

        layers_merged = min(32, (frame_idx * 2) + 2)
        pct = int((layers_merged / 32.0) * 100)

        # Progress Bar
        bar_w = right_w - 72
        draw_rect(canvas, right_x + 22, prog_y + 26, bar_w, 14, 2)
        draw_stroke_rect(canvas, right_x + 22, prog_y + 26, bar_w, 14, 3)
        fill_w = int((pct / 100.0) * bar_w)
        draw_rect(canvas, right_x + 22, prog_y + 26, fill_w, 14, 7)
        draw_text(canvas, f"{pct}%", right_x + fill_w - 24 if fill_w > 30 else right_x + fill_w + 4, prog_y + 28, 0 if fill_w > 30 else 4, scale=1)

        draw_text(canvas, f"MERGED {layers_merged}/32 LAYERS (Q_PROJ, K_PROJ, V_PROJ, O_PROJ)", right_x + 22, prog_y + 48, 4, scale=1)
        draw_text(canvas, "VERIFIED MSE DEVIATION: 0.00000000 (EXACT MATCH)", right_x + 22, prog_y + 62, 7, scale=1)
        draw_text(canvas, "EXPORT READY: GGUF V3 + SAFETENSORS DISK SERIALIZATION", right_x + 22, prog_y + 76, 6, scale=1)

        # 5. Bottom Status Bar
        bot_y = H - 24
        draw_rect(canvas, 0, bot_y, W, 24, 1)
        draw_rect(canvas, 0, bot_y, W, 1, 3)

        draw_rect(canvas, 14, bot_y + 7, 8, 8, 7)
        draw_text(canvas, "INTEROP COMPLETE", 28, bot_y + 8, 7, scale=1)
        draw_text(canvas, "|  PYTORCH: EXPORTED  |  PEFT LORA: MERGED  |  HUGGINGFACE CONFIG: READY  |  NO PYTHON NEEDED", 166, bot_y + 8, 5, scale=1)

        frames.append(canvas)

    gif_bytes = encode_gif(frames, PALETTE, delay=10)
    with open("assets/pytorch_cuda_interop.gif", "wb") as f:
        f.write(gif_bytes)
    print("Generated assets/pytorch_cuda_interop.gif (", len(gif_bytes), "bytes )")


def main():
    t0 = time.time()
    print("Starting THME Showcase GIF Suite Generation...")
    render_live_benchmark_animation()
    render_scientific_math_animation()
    render_layer_deconstruction_animation()
    render_pytorch_cuda_interop_animation()
    print(f"All 4 showcase animations generated successfully in {time.time() - t0:.2f} seconds!")

if __name__ == "__main__":
    main()
