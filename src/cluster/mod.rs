// SPDX-License-Identifier: AGPL-3.0-only
//! Heterogeneous Multi-PC Clustering over Wi-Fi, Ethernet & Thunderbolt (`THME_Research_Dossier_v3.3` §7.12).

use crate::engine::error::{EngineError, EngineResult};

/// Probed interconnect tier between two cluster nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterconnectTier {
    /// `3-4 GB/s`, `< 20 us` RTT: supports Tensor-Parallel (`TP`) + Pipeline (`PP`).
    Thunderbolt4,
    /// `0.12-1.2 GB/s`, `0.1-0.5 ms` RTT: coarse Pipeline (`PP`) + `Q4_1` wire.
    Ethernet,
    /// `0.05-0.3 GB/s`, `2-15 ms` jitter: zero synchronous `TP`; Edge-Draft / Verify (`32 B` tokens).
    Wifi,
}

/// Allowed parallelism mode for a given interconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterParallelMode {
    TensorAndPipeline,
    CoarsePipelineQ41,
    AsyncEdgeDraftVerify,
}

/// Classifies the interconnect tier and prohibits synchronous `TP` if `rtt_ms > 0.5` (`R131`).
pub fn classify_interconnect(
    bandwidth_gbps: f64,
    rtt_ms: f64,
) -> (InterconnectTier, ClusterParallelMode) {
    if bandwidth_gbps >= 2.0 && rtt_ms <= 0.05 {
        (
            InterconnectTier::Thunderbolt4,
            ClusterParallelMode::TensorAndPipeline,
        )
    } else if rtt_ms <= 0.5 && bandwidth_gbps >= 0.1 {
        (
            InterconnectTier::Ethernet,
            ClusterParallelMode::CoarsePipelineQ41,
        )
    } else {
        (
            InterconnectTier::Wifi,
            ClusterParallelMode::AsyncEdgeDraftVerify,
        )
    }
}

/// Heterogeneous compute node capability.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub id: String,
    pub mem_bandwidth_gbps: f64,
    pub avg_bpw: f64,
    pub vram_avail_bytes: u64,
}

/// Assigned contiguous layer stage `[start_layer, end_layer)` for a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageAssignment {
    pub node_id: String,
    pub start_layer: usize,
    pub end_layer: usize,
}

/// Partitions `total_layers` contiguously across surviving `nodes` proportional
/// to `mem_bandwidth_gbps / avg_bpw` (`R133` fault-tolerant stage re-routing).
pub fn partition_stages(
    nodes: &[ClusterNode],
    total_layers: usize,
) -> EngineResult<Vec<StageAssignment>> {
    if nodes.is_empty() || total_layers == 0 {
        return Err(EngineError::InvalidInput(
            "partition_stages requires non-empty nodes and total_layers > 0".to_string(),
        ));
    }
    let mut scores = Vec::with_capacity(nodes.len());
    let mut sum_score = 0.0_f64;
    for node in nodes {
        if !node.mem_bandwidth_gbps.is_finite()
            || node.mem_bandwidth_gbps <= 0.0
            || !node.avg_bpw.is_finite()
            || node.avg_bpw <= 0.0
        {
            return Err(EngineError::InvalidInput(format!(
                "invalid capability on node '{}'",
                node.id
            )));
        }
        let s = node.mem_bandwidth_gbps / node.avg_bpw;
        scores.push(s);
        sum_score += s;
    }

    let mut out = Vec::with_capacity(nodes.len());
    let mut cursor = 0_usize;
    for (idx, node) in nodes.iter().enumerate() {
        let end = if idx + 1 == nodes.len() {
            total_layers
        } else {
            let count = ((scores[idx] / sum_score) * (total_layers as f64)).round() as usize;
            (cursor + count).min(total_layers)
        };
        out.push(StageAssignment {
            node_id: node.id.clone(),
            start_layer: cursor,
            end_layer: end,
        });
        cursor = end;
    }
    Ok(out)
}

/// Quantizes a 32-element stage-boundary activation block to `Q4_1` wire format
/// (`scale`, `min`, `16` packed nibble bytes = `0.5625` B/elem) (`R136`).
pub fn pack_q4_1_block32(vals: &[f32; 32]) -> (f32, f32, [u8; 16]) {
    let min_v = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = ((max_v - min_v) / 15.0).max(1e-8);
    let mut packed = [0_u8; 16];
    for (i, &v) in vals.iter().enumerate() {
        let q = (((v - min_v) / scale).round()).clamp(0.0, 15.0) as u8;
        if i.is_multiple_of(2) {
            packed[i / 2] |= q & 0x0F;
        } else {
            packed[i / 2] |= (q & 0x0F) << 4;
        }
    }
    (scale, min_v, packed)
}

/// Dequantizes a 32-element `Q4_1` stage-boundary wire block back to `FP32`.
pub fn unpack_q4_1_block32(scale: f32, min_v: f32, packed: &[u8; 16]) -> [f32; 32] {
    let mut out = [0.0_f32; 32];
    for (i, dst) in out.iter_mut().enumerate() {
        let q = if i.is_multiple_of(2) {
            packed[i / 2] & 0x0F
        } else {
            (packed[i / 2] >> 4) & 0x0F
        };
        *dst = min_v + f32::from(q) * scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wifi_prohibits_synchronous_tensor_parallel() {
        let (tier, mode) = classify_interconnect(0.25, 8.0);
        assert_eq!(tier, InterconnectTier::Wifi);
        assert_eq!(mode, ClusterParallelMode::AsyncEdgeDraftVerify);
    }

    #[test]
    fn stage_partition_covers_all_layers_proportionally() {
        let nodes = vec![
            ClusterNode {
                id: "desktop-gpu".into(),
                mem_bandwidth_gbps: 288.0,
                avg_bpw: 4.5,
                vram_avail_bytes: 14_000_000_000,
            },
            ClusterNode {
                id: "laptop-bitcos".into(),
                mem_bandwidth_gbps: 96.0,
                avg_bpw: 1.61,
                vram_avail_bytes: 6_000_000_000,
            },
        ];
        let plan = partition_stages(&nodes, 24).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].start_layer, 0);
        assert_eq!(plan[0].end_layer, plan[1].start_layer);
        assert_eq!(plan[1].end_layer, 24);
    }

    #[test]
    fn q4_1_wire_round_trip_bounds_error() {
        let mut vals = [0.0_f32; 32];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 1.5;
        }
        let (s, m, p) = pack_q4_1_block32(&vals);
        let dec = unpack_q4_1_block32(s, m, &p);
        for (a, b) in vals.iter().zip(dec.iter()) {
            assert!((a - b).abs() <= s * 0.55);
        }
    }
}
