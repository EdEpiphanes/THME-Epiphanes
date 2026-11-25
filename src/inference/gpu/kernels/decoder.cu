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
    __shared__ float smem[256];
    unsigned int h = blockIdx.x;
    unsigned int t = blockIdx.y;
    unsigned int tid = threadIdx.x;
    if (h >= heads || t >= t_len) {
        return;
    }

    unsigned int base = (h * t_len + t) * t_len;
    float lm = -3.402823466e+38f;
    for (unsigned int s = tid; s <= t; s += 256u) {
        lm = fmaxf(lm, scores[base + s] * scale);
    }
    smem[tid] = lm;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float m = smem[0];
    __syncthreads();

    float ls = 0.0f;
    for (unsigned int s = tid; s <= t; s += 256u) {
        ls += expf(scores[base + s] * scale - m);
    }
    smem[tid] = ls;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
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
