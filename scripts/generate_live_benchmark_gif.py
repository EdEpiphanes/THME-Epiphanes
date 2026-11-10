import struct
import numpy as np

# 5x7 ASCII font dictionary for clean rasterization
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
    '0': [0x1C, 0x26, 0x2A, 0x32, 0x22, 0x22, 0x1C],
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
    '-': [0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00],
    '+': [0x00, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x00],
    '[': [0x1E, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1E],
    ']': [0x3C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x3C],
    '/': [0x02, 0x04, 0x08, 0x10, 0x20, 0x00, 0x00],
    '%': [0x32, 0x34, 0x08, 0x10, 0x12, 0x26, 0x00],
    '>': [0x20, 0x10, 0x08, 0x04, 0x08, 0x10, 0x20],
    '<': [0x04, 0x08, 0x10, 0x20, 0x10, 0x08, 0x04],
}

PALETTE = [
    (11, 15, 25),    # 0: BG Dark
    (21, 30, 46),    # 1: Card BG
    (30, 41, 59),    # 2: Input / Box BG
    (51, 65, 85),    # 3: Border
    (248, 250, 252), # 4: Text Main
    (148, 163, 184), # 5: Text Muted
    (6, 182, 212),   # 6: Cyan (Active token / accent)
    (16, 185, 129),  # 7: Emerald (Exit gate / Speed)
    (245, 158, 11),  # 8: Amber (KV Cache)
    (244, 63, 94),   # 9: Rose (Latency spike)
    (168, 85, 247),  # 10: Purple (MLP)
    (59, 130, 246),  # 11: Blue (Transformer)
    (255, 255, 255), # 12: Bright White
    (14, 165, 233),  # 13: Cyan glow
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

def encode_gif(frames, palette, delay=12):
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

def render_benchmark_animation():
    W, H = 840, 440
    num_frames = 24
    frames = []

    words = ["MANIFOLD", "TOPOLOGY", "QUANTIZE", "THROUGHPUT", "LATENCY", "BITNET", "DELTANET", "PARETO"]

    for frame_idx in range(num_frames):
        canvas = np.zeros((H, W), dtype=np.uint8)

        # 1. Header Bar
        draw_rect(canvas, 0, 0, W, 48, 1) # Card BG
        draw_rect(canvas, 14, 12, 80, 24, 6) # Cyan Badge
        draw_text(canvas, "THME V2", 24, 20, 0, scale=1)
        draw_text(canvas, "THME STUDIO -- EPIPHANES LIVE NEURAL BENCHMARK & TOKEN FLOW", 108, 16, 4, scale=1)
        draw_text(canvas, "LLAMA-3-8B [GGUF/SAFETENSORS] -- 32 LAYERS -- 8192 CONTEXT -- FP16/INT4 HYBRID", 108, 30, 5, scale=1)

        # 2. Stats Ribbon
        draw_rect(canvas, 0, 50, W, 40, 0) # Dark BG
        box_w = (W - 28) // 4
        stats = [
            ("THROUGHPUT", f"{42.4 + (frame_idx % 5)*1.2:.1f} TOK/S", 6),
            ("MEMORY BANDWIDTH", f"{378.2 + (frame_idx % 4)*14.1:.1f} GB/S", 7),
            ("KV CACHE FOOTPRINT", f"{142.6 + frame_idx*1.8:.1f} MB", 8),
            ("EXIT SPECULATION", "80% / 20% (L8)", 7),
        ]
        for i, (title, val, col) in enumerate(stats):
            bx = 14 + i * box_w
            draw_rect(canvas, bx, 52, box_w - 8, 36, 1)
            draw_stroke_rect(canvas, bx, 52, box_w - 8, 36, 3)
            draw_text(canvas, title, bx + 8, 56, 5, scale=1)
            draw_text(canvas, val, bx + 8, 70, col, scale=1)

        # 3. Left Panel: Live Token Terminal
        left_w = 340
        draw_rect(canvas, 14, 98, left_w, 328, 1)
        draw_stroke_rect(canvas, 14, 98, left_w, 328, 3)
        draw_rect(canvas, 14, 98, left_w, 28, 2)
        draw_text(canvas, ">_ AUTOREGRESSIVE TOKEN STREAM", 24, 108, 6, scale=1)

        prompt_lines = [
            "PROMPT: EXPLAIN HOW HETEROGENEOUS",
            "REPRESENTATIONS REDUCE MEMORY BW.",
            "",
            "GENERATION:",
        ]
        for l_idx, pl in enumerate(prompt_lines):
            draw_text(canvas, pl, 24, 136 + l_idx * 16, 5 if l_idx < 3 else 4, scale=1)

        # Stream words based on frame
        word_count = min(len(words), (frame_idx // 3) + 1)
        active_stream = " ".join(words[:word_count])
        # Wrap into 2 lines
        w1 = active_stream[:36]
        w2 = active_stream[36:]
        draw_text(canvas, ">> " + w1, 24, 204, 6, scale=1)
        if w2:
            draw_text(canvas, "   " + w2, 24, 222, 6, scale=1)

        # Active status indicator
        draw_rect(canvas, 24, 260, left_w - 20, 32, 2)
        draw_stroke_rect(canvas, 24, 260, left_w - 20, 32, 6)
        draw_text(canvas, f"STATUS: GENERATING TOKEN {word_count}/32", 34, 268, 7, scale=1)
        draw_text(canvas, f"STAGE: ATTENTION KV CACHE APPEND", 34, 280, 6, scale=1)

        # Progress bar
        p_pct = int(((frame_idx + 1) / num_frames) * (left_w - 30))
        draw_rect(canvas, 24, 304, left_w - 20, 8, 3)
        draw_rect(canvas, 24, 304, p_pct, 8, 6)

        # 4. Right Panel: Live Neural Pipeline
        right_x = 14 + left_w + 12
        right_w = W - right_x - 14
        draw_rect(canvas, right_x, 98, right_w, 328, 1)
        draw_stroke_rect(canvas, right_x, 98, right_w, 328, 3)
        draw_rect(canvas, right_x, 98, right_w, 28, 2)
        draw_text(canvas, "TRANSFORMER PIPELINE & SPECULATIVE EXIT FLOW", right_x + 10, 108, 4, scale=1)

        # Draw Neural Flow Backbone
        spine_y = 175
        draw_rect(canvas, right_x + 20, spine_y - 1, right_w - 40, 3, 3)

        # Nodes: IN, L0, L1, L2, L3, L4, L5, L6 (Exit), L7, L8, L9, OUT
        layers = ["IN", "L0", "L1", "L2", "L3", "L4", "L5", "L6*", "L7", "L8", "L9", "OUT"]
        node_step = (right_w - 60) // (len(layers) - 1)

        active_layer_idx = (frame_idx % (len(layers) - 1)) + 1
        pulse_x = right_x + 30 + (frame_idx / num_frames) * (right_w - 60)

        for idx, l_name in enumerate(layers):
            nx = right_x + 30 + idx * node_step
            is_active = (idx == active_layer_idx)
            is_exit = ("*" in l_name)
            is_endpoint = (idx == 0 or idx == len(layers) - 1)

            box_size = 28
            bx = nx - box_size // 2
            by = spine_y - box_size // 2

            bg_col = 6 if is_active else (7 if is_exit else (2 if not is_endpoint else 11))
            border_col = 12 if is_active else (7 if is_exit else 3)

            draw_rect(canvas, bx, by, box_size, box_size, bg_col)
            draw_stroke_rect(canvas, bx, by, box_size, box_size, border_col, thickness=2 if is_active else 1)
            draw_text(canvas, l_name[:3], bx + 4, by + 10, 0 if is_active else 4, scale=1)

            # Exit checkpoint branch
            if is_exit:
                draw_rect(canvas, nx - 1, by - 24, 2, 24, 7)
                draw_rect(canvas, nx - 22, by - 36, 44, 14, 7)
                draw_text(canvas, "80% EXIT", nx - 20, by - 33, 0, scale=1)

        # Glowing Pulse Particle
        px = int(pulse_x)
        draw_rect(canvas, px - 6, spine_y - 6, 12, 12, 13)
        draw_rect(canvas, px - 3, spine_y - 3, 6, 6, 12)

        # Sub-panel: Layer Latency Waterfall
        wf_y = 230
        draw_rect(canvas, right_x + 14, wf_y, right_w - 28, 180, 2)
        draw_stroke_rect(canvas, right_x + 14, wf_y, right_w - 28, 180, 3)
        draw_text(canvas, "LAYER LATENCY WATERFALL (MS) & QUANTIZATION PROFILE", right_x + 24, wf_y + 10, 4, scale=1)

        num_wf_bars = 8
        bar_step_y = 18
        for b in range(num_wf_bars):
            by = wf_y + 32 + b * bar_step_y
            is_cur_b = (b == (active_layer_idx % num_wf_bars))
            draw_text(canvas, f"L{b}", right_x + 24, by + 2, 6 if is_cur_b else 5, scale=1)

            bw = 140 + int(np.sin(frame_idx * 0.4 + b) * 35) + (40 if is_cur_b else 0)
            bar_col = 6 if is_cur_b else (10 if b % 2 == 1 else 11)
            draw_rect(canvas, right_x + 48, by, bw, 12, bar_col)

            lat_val = f"{1.8 + (bw / 100.0):.2f}MS"
            draw_text(canvas, lat_val, right_x + 56 + bw, by + 2, 12 if is_cur_b else 4, scale=1)
            draw_text(canvas, "Q4_K / TERNARY" if b > 1 else "FP16 DENSE", right_x + right_w - 140, by + 2, 5, scale=1)

        frames.append(canvas)

    gif_bytes = encode_gif(frames, PALETTE, delay=12)
    with open("assets/live_neural_benchmark.gif", "wb") as f:
        f.write(gif_bytes)
    print("Successfully generated assets/live_neural_benchmark.gif, size:", len(gif_bytes))

if __name__ == "__main__":
    render_benchmark_animation()

