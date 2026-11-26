// SPDX-License-Identifier: AGPL-3.0-only
// Header-free CUDA kernels for the device-resident LLaMA-style decoder.
// Uses built-in types and inline PTX for f32->f16 so NVRTC needs no external include paths.

extern "C" __global__ void f32_to_f16(
    const float* in,
    unsigned short* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out[i]) : "f"(in[i]));
    }
}

extern "C" __global__ void rms_norm(
    const float* x,
    const float* w,
    float* out,
    unsigned int d,
    float eps,
    unsigned int x_offset
) {
    __shared__ float smem[256];
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int off = x_offset + row * d;

    float ss = 0.0f;
    for (unsigned int i = tid; i < d; i += 256) {
        float v = x[off + i];
        ss += v * v;
    }
    smem[tid] = ss;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(smem[0] / (float)d + eps);
    for (unsigned int i = tid; i < d; i += 256) {
        out[row * d + i] = x[off + i] * scale * w[i];
    }
}

extern "C" __global__ void rms_norm_f16(
    const float* x,
    const float* w,
    unsigned short* out16,
    unsigned int d,
    float eps,
    unsigned int x_offset
) {
    __shared__ float smem[256];
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int off = x_offset + row * d;

    float ss = 0.0f;
    for (unsigned int i = tid; i < d; i += 256) {
        float v = x[off + i];
        ss += v * v;
    }
    smem[tid] = ss;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(smem[0] / (float)d + eps);
    for (unsigned int i = tid; i < d; i += 256) {
        float val = x[off + i] * scale * w[i];
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(val));
        out16[row * d + i] = h;
    }
}

extern "C" __global__ void rope(
    const float* cos_tbl,
    const float* sin_tbl,
    float* qk,
    unsigned int rows,
    unsigned int row_len,
    unsigned int head_dim,
    unsigned int half
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int heads = row_len / head_dim;
    unsigned int per_row = heads * half;
    if (idx >= rows * per_row) {
        return;
    }
    unsigned int row = idx / per_row;
    unsigned int r = idx % per_row;
    unsigned int head = r / half;
    unsigned int j = r % half;
    unsigned int base = row * row_len + head * head_dim;

    float a = qk[base + j];
    float b = qk[base + half + j];
    float c = cos_tbl[row * half + j];
    float s = sin_tbl[row * half + j];
    qk[base + j] = a * c - b * s;
    qk[base + half + j] = b * c + a * s;
}

extern "C" __global__ void causal_softmax_f16(
    const float* scores,
    unsigned short* probs16,
    unsigned int t_len,
    unsigned int heads,
    float scale
) {
    __shared__ float smem[8];
    unsigned int h = blockIdx.x;
    unsigned int t = blockIdx.y;
    unsigned int tid = threadIdx.x;
    if (h >= heads || t >= t_len) {
        return;
    }
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    unsigned int base = (h * t_len + t) * t_len;

    float lm = -3.402823466e+38f;
    for (unsigned int s = tid; s <= t; s += 256u) {
        lm = fmaxf(lm, scores[base + s] * scale);
    }
    for (int off = 16; off > 0; off >>= 1) {
        lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, off));
    }
    if (lane == 0u) { smem[warp] = lm; }
    __syncthreads();
    if (warp == 0u) {
        float vm = (lane < 8u) ? smem[lane] : -3.402823466e+38f;
        for (int off = 4; off > 0; off >>= 1) {
            vm = fmaxf(vm, __shfl_xor_sync(0xffffffffu, vm, off));
        }
        if (lane == 0u) { smem[0] = vm; }
    }
    __syncthreads();
    float m = smem[0];

    float ls = 0.0f;
    for (unsigned int s = tid; s <= t; s += 256u) {
        ls += expf(scores[base + s] * scale - m);
    }
    for (int off = 16; off > 0; off >>= 1) {
        ls += __shfl_xor_sync(0xffffffffu, ls, off);
    }
    if (lane == 0u) { smem[warp] = ls; }
    __syncthreads();
    if (warp == 0u) {
        float vs = (lane < 8u) ? smem[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            vs += __shfl_xor_sync(0xffffffffu, vs, off);
        }
        if (lane == 0u) { smem[0] = vs; }
    }
    __syncthreads();
    float inv_l = 1.0f / smem[0];

    for (unsigned int s = tid; s < t_len; s += 256u) {
        float p = (s <= t) ? (expf(scores[base + s] * scale - m) * inv_l) : 0.0f;
        unsigned short hval;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hval) : "f"(p));
        probs16[base + s] = hval;
    }
}

extern "C" __global__ void swiglu(
    const float* gate,
    const float* up,
    float* out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        out[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

extern "C" __global__ void swiglu_f16(
    const float* gate,
    const float* up,
    unsigned short* out16,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float val = (g / (1.0f + expf(-g))) * up[idx];
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(val));
        out16[idx] = h;
    }
}

extern "C" __global__ void add(
    const float* a,
    const float* b,
    float* out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void nll_f16(
    const unsigned short* logits16,
    const unsigned int* targets,
    float* out,
    unsigned int vocab,
    unsigned int first_row
) {
    __shared__ float smem[8];
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    unsigned int base = row * vocab;

    float lm = -3.402823466e+38f;
    for (unsigned int i = tid; i < vocab; i += 256u) {
        float v;
        asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(logits16[base + i]));
        lm = fmaxf(lm, v);
    }
    for (int off = 16; off > 0; off >>= 1) {
        lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, off));
    }
    if (lane == 0u) { smem[warp] = lm; }
    __syncthreads();
    if (warp == 0u) {
        float vm = (lane < 8u) ? smem[lane] : -3.402823466e+38f;
        for (int off = 4; off > 0; off >>= 1) {
            vm = fmaxf(vm, __shfl_xor_sync(0xffffffffu, vm, off));
        }
        if (lane == 0u) { smem[0] = vm; }
    }
    __syncthreads();
    float m = smem[0];

    float ls = 0.0f;
    for (unsigned int i = tid; i < vocab; i += 256u) {
        float v;
        asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(logits16[base + i]));
        ls += expf(v - m);
    }
    for (int off = 16; off > 0; off >>= 1) {
        ls += __shfl_xor_sync(0xffffffffu, ls, off);
    }
    if (lane == 0u) { smem[warp] = ls; }
    __syncthreads();
    if (warp == 0u) {
        float vs = (lane < 8u) ? smem[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            vs += __shfl_xor_sync(0xffffffffu, vs, off);
        }
        if (lane == 0u) { smem[0] = vs; }
    }
    __syncthreads();

    if (tid == 0u) {
        unsigned int g = first_row + row;
        unsigned int target_id = targets[g];
        float vt;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vt) : "h"(logits16[base + target_id]));
        out[g] = m + logf(smem[0]) - vt;
    }
}

extern "C" __global__ void gemm_f16_tiled(
    const unsigned short* x16,
    const unsigned short* w16,
    float* out_f32,
    unsigned short* out_f16,
    unsigned int rows,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int x_row_off,
    unsigned int write_f16
) {
    __shared__ float xs[16][16];
    __shared__ float ws[16][16];
    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    float acc = 0.0f;

    for (unsigned int k0 = 0u; k0 < in_dim; k0 += 16u) {
        unsigned int xk = k0 + threadIdx.x;
        unsigned int wk = k0 + threadIdx.y;
        float xv = 0.0f;
        float wv = 0.0f;
        if (row < rows && xk < in_dim) {
            asm("cvt.f32.f16 %0, %1;" : "=f"(xv) : "h"(x16[(x_row_off + row) * in_dim + xk]));
        }
        if (col < out_dim && wk < in_dim) {
            asm("cvt.f32.f16 %0, %1;" : "=f"(wv) : "h"(w16[col * in_dim + wk]));
        }
        xs[threadIdx.y][threadIdx.x] = xv;
        ws[threadIdx.y][threadIdx.x] = wv;
        __syncthreads();
        for (unsigned int k = 0u; k < 16u; ++k) {
            acc += xs[threadIdx.y][k] * ws[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && col < out_dim) {
        unsigned int idx = row * out_dim + col;
        if (write_f16 != 0u) {
            unsigned short h;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(acc));
            out_f16[idx] = h;
        } else {
            out_f32[idx] = acc;
        }
    }
}

extern "C" __global__ void causal_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    unsigned int t_len,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    float scale
) {
    unsigned int h = blockIdx.x;
    unsigned int t = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int lane = threadIdx.x;
    if (h >= heads || t >= t_len) { return; }

    unsigned int d = heads * head_dim;
    unsigned int kv_dim = kv_heads * head_dim;
    unsigned int kv_off = (h / (heads / kv_heads)) * head_dim;
    unsigned int q_base = t * d + h * head_dim;
    unsigned int elems = (head_dim + 31u) >> 5;

    float qr[8];
    float acc[8];
    for (unsigned int i = 0; i < elems; ++i) {
        unsigned int idx = lane + i * 32u;
        qr[i] = (idx < head_dim) ? q[q_base + idx] : 0.0f;
        acc[i] = 0.0f;
    }

    float m = -3.402823466e+38f;
    float l = 0.0f;
    for (unsigned int s = 0; s <= t; ++s) {
        unsigned int k_base = s * kv_dim + kv_off;
        float dot = 0.0f;
        for (unsigned int i = 0; i < elems; ++i) {
            unsigned int idx = lane + i * 32u;
            if (idx < head_dim) { dot += qr[i] * k[k_base + idx]; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }
        float sc = dot * scale;
        float m_new = fmaxf(m, sc);
        float alpha = (s == 0u) ? 0.0f : expf(m - m_new);
        float beta = expf(sc - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (unsigned int i = 0; i < elems; ++i) {
            unsigned int idx = lane + i * 32u;
            if (idx < head_dim) { acc[i] = acc[i] * alpha + beta * v[k_base + idx]; }
        }
    }

    float inv_l = 1.0f / l;
    for (unsigned int i = 0; i < elems; ++i) {
        unsigned int idx = lane + i * 32u;
        if (idx < head_dim) { out[q_base + idx] = acc[i] * inv_l; }
    }
}

extern "C" __global__ void nll(
    const float* logits,
    const unsigned int* targets,
    float* out,
    unsigned int vocab,
    unsigned int first_row
) {
    __shared__ float smem[256];
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int base = row * vocab;

    float lm = -3.402823466e+38f;
    for (unsigned int i = tid; i < vocab; i += 256) {
        lm = fmaxf(lm, logits[base + i]);
    }
    smem[tid] = lm;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    float m = smem[0];
    __syncthreads();

    float ls = 0.0f;
    for (unsigned int i = tid; i < vocab; i += 256) {
        ls += expf(logits[base + i] - m);
    }
    smem[tid] = ls;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float total = smem[0];

    if (tid == 0) {
        unsigned int g = first_row + row;
        unsigned int target_id = targets[g];
        out[g] = m + logf(total) - logits[base + target_id];
    }
}

extern "C" __global__ void rope_qk_f16(
    const float* cos_tbl,
    const float* sin_tbl,
    unsigned short* q16,
    unsigned short* k16,
    unsigned int rows,
    unsigned int q_dim,
    unsigned int kv_dim,
    unsigned int head_dim,
    unsigned int half
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int q_per_row = (q_dim / head_dim) * half;
    unsigned int k_per_row = (kv_dim / head_dim) * half;
    unsigned int total_q = rows * q_per_row;
    unsigned int total_k = rows * k_per_row;
    if (idx >= total_q + total_k) {
        return;
    }

    unsigned short* ptr;
    unsigned int row_len;
    unsigned int local_idx;
    unsigned int per_row;
    if (idx < total_q) {
        ptr = q16;
        row_len = q_dim;
        local_idx = idx;
        per_row = q_per_row;
    } else {
        ptr = k16;
        row_len = kv_dim;
        local_idx = idx - total_q;
        per_row = k_per_row;
    }

    unsigned int row = local_idx / per_row;
    unsigned int r = local_idx % per_row;
    unsigned int head = r / half;
    unsigned int j = r % half;
    unsigned int base = row * row_len + head * head_dim;

    float a, b;
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(ptr[base + j]));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(ptr[base + half + j]));
    float c = cos_tbl[row * half + j];
    float s = sin_tbl[row * half + j];

    unsigned short ha, hb;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(ha) : "f"(a * c - b * s));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hb) : "f"(b * c + a * s));
    ptr[base + j] = ha;
    ptr[base + half + j] = hb;
}

extern "C" __global__ void swiglu_f16_in(
    const unsigned short* gate16,
    const unsigned short* up16,
    unsigned short* out16,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g, u;
        asm("cvt.f32.f16 %0, %1;" : "=f"(g) : "h"(gate16[idx]));
        asm("cvt.f32.f16 %0, %1;" : "=f"(u) : "h"(up16[idx]));
        float val = (g / (1.0f + expf(-g))) * u;
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(val));
        out16[idx] = h;
    }
}

extern "C" __global__ void add_rms_norm_f16(
    const float* x_in,
    const unsigned short* res16,
    const float* w,
    float* x_out,
    unsigned short* out16,
    unsigned int d,
    float eps
) {
    __shared__ float smem[8];
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    unsigned int off = row * d;

    float ss = 0.0f;
    for (unsigned int i = tid; i < d; i += 256u) {
        float r;
        asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(res16[off + i]));
        float v = x_in[off + i] + r;
        x_out[off + i] = v;
        ss += v * v;
    }
    for (int s = 16; s > 0; s >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, s);
    }
    if (lane == 0u) { smem[warp] = ss; }
    __syncthreads();
    if (warp == 0u) {
        float vs = (lane < 8u) ? smem[lane] : 0.0f;
        for (int s = 4; s > 0; s >>= 1) {
            vs += __shfl_xor_sync(0xffffffffu, vs, s);
        }
        if (lane == 0u) { smem[0] = vs; }
    }
    __syncthreads();

    float scale = rsqrtf(smem[0] / (float)d + eps);
    for (unsigned int i = tid; i < d; i += 256u) {
        float val = x_out[off + i] * scale * w[i];
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(val));
        out16[off + i] = h;
    }
}

extern "C" __global__ void flash_attention_f16(
    const unsigned short* q16,
    const unsigned short* k16,
    const unsigned short* v16,
    unsigned short* out16,
    unsigned int t_len,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    float scale
) {
    // 8 warps per block: each warp handles one query row t = t0 + warp_id.
    // Shared memory caches a 32-row tile of K and V for the shared KV head.
    extern __shared__ unsigned short kv_smem[];
    unsigned short* k_tile = kv_smem;
    unsigned short* v_tile = kv_smem + 32u * head_dim;

    unsigned int h = blockIdx.x;
    unsigned int t0 = blockIdx.y * 8u;
    unsigned int lane = threadIdx.x;
    unsigned int warp = threadIdx.y;
    unsigned int tid = warp * 32u + lane;
    unsigned int t = t0 + warp;
    if (h >= heads || t0 >= t_len) {
        return;
    }

    unsigned int group = heads / kv_heads;
    unsigned int kvh = h / group;
    unsigned int q_stride = heads * head_dim;
    unsigned int kv_stride = kv_heads * head_dim;
    unsigned int t_max = t0 + 7u;
    if (t_max >= t_len) {
        t_max = t_len - 1u;
    }

    float q_reg[8];
    float acc[8];
    for (unsigned int r = 0u; r < 8u; ++r) {
        q_reg[r] = 0.0f;
        acc[r] = 0.0f;
    }
    if (t < t_len) {
        unsigned int q_off = t * q_stride + h * head_dim;
        for (unsigned int d = lane, r = 0u; d < head_dim && r < 8u; d += 32u, ++r) {
            float qv;
            asm("cvt.f32.f16 %0, %1;" : "=f"(qv) : "h"(q16[q_off + d]));
            q_reg[r] = qv * scale;
        }
    }

    float m_i = -3.402823466e+38f;
    float l_i = 0.0f;
    unsigned int tile_elems = 32u * head_dim;

    for (unsigned int kb = 0u; kb <= t_max; kb += 32u) {
        for (unsigned int idx = tid; idx < tile_elems; idx += 256u) {
            unsigned int row = idx / head_dim;
            unsigned int col = idx % head_dim;
            unsigned int ks = kb + row;
            if (ks < t_len) {
                unsigned int kv_off = ks * kv_stride + kvh * head_dim + col;
                k_tile[idx] = k16[kv_off];
                v_tile[idx] = v16[kv_off];
            } else {
                k_tile[idx] = 0;
                v_tile[idx] = 0;
            }
        }
        __syncthreads();

        if (t < t_len) {
            unsigned int limit = 32u;
            if (kb + limit - 1u > t) {
                limit = t - kb + 1u;
            }
            for (unsigned int j = 0u; j < limit; ++j) {
                unsigned int j_off = j * head_dim;
                float dot = 0.0f;
                for (unsigned int d = lane, r = 0u; d < head_dim && r < 8u; d += 32u, ++r) {
                    float kv;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(kv) : "h"(k_tile[j_off + d]));
                    dot += q_reg[r] * kv;
                }
                for (int off = 16; off > 0; off >>= 1) {
                    dot += __shfl_xor_sync(0xffffffffu, dot, off);
                }

                float m_new = fmaxf(m_i, dot);
                float alpha = expf(m_i - m_new);
                float p = expf(dot - m_new);
                l_i = l_i * alpha + p;
                m_i = m_new;

                for (unsigned int d = lane, r = 0u; d < head_dim && r < 8u; d += 32u, ++r) {
                    float vv;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(vv) : "h"(v_tile[j_off + d]));
                    acc[r] = acc[r] * alpha + p * vv;
                }
            }
        }
        __syncthreads();
    }

    if (t < t_len) {
        float inv_l = 1.0f / l_i;
        unsigned int o_off = t * q_stride + h * head_dim;
        for (unsigned int d = lane, r = 0u; d < head_dim && r < 8u; d += 32u, ++r) {
            unsigned short hv;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(acc[r] * inv_l));
            out16[o_off + d] = hv;
        }
    }
}
