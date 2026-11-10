use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheProjection {
    pub context_length: usize,
    pub batch_size: usize,
    pub bytes_fp16: usize,
    pub bytes_fp8: usize,
    pub bytes_q4: usize,
}

impl KvCacheProjection {
    pub fn mb_fp16(&self) -> f64 {
        self.bytes_fp16 as f64 / (1024.0 * 1024.0)
    }

    pub fn mb_fp8(&self) -> f64 {
        self.bytes_fp8 as f64 / (1024.0 * 1024.0)
    }

    pub fn mb_q4(&self) -> f64 {
        self.bytes_q4 as f64 / (1024.0 * 1024.0)
    }
}

pub struct KvCacheEstimator;

impl KvCacheEstimator {
    /// Formula: M_KV = 2 * L * B * S * H_KV * D_head * bytes_per_element
    /// Factor of 2 represents Key and Value tensors
    pub fn calculate_bytes(
        num_layers: usize,
        batch_size: usize,
        context_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        bytes_per_elem: f64,
    ) -> usize {
        let elements = 2.0
            * (num_layers as f64)
            * (batch_size as f64)
            * (context_len as f64)
            * (num_kv_heads as f64)
            * (head_dim as f64);
        (elements * bytes_per_elem).ceil() as usize
    }

    /// Generate KV Cache memory curve across context lengths
    pub fn project_memory_curve(
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        context_lengths: &[usize],
    ) -> Vec<KvCacheProjection> {
        context_lengths
            .iter()
            .map(|&s| {
                let bytes_fp16 =
                    Self::calculate_bytes(num_layers, batch_size, s, num_kv_heads, head_dim, 2.0);
                let bytes_fp8 =
                    Self::calculate_bytes(num_layers, batch_size, s, num_kv_heads, head_dim, 1.0);
                let bytes_q4 =
                    Self::calculate_bytes(num_layers, batch_size, s, num_kv_heads, head_dim, 0.5);

                KvCacheProjection {
                    context_length: s,
                    batch_size,
                    bytes_fp16,
                    bytes_fp8,
                    bytes_q4,
                }
            })
            .collect()
    }
}
