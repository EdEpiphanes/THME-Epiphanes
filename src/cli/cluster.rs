// SPDX-License-Identifier: AGPL-3.0-only
//! `thme cluster`: heterogeneous multi-PC interconnect classification & stage partitioning (§7.12).

use anyhow::{bail, Result};
use clap::Args;

use crate::cluster::{classify_interconnect, partition_stages, ClusterNode, ClusterParallelMode};

#[derive(Args, Debug)]
pub struct ClusterArgs {
    /// Measured interconnect bandwidth in GB/s (e.g. 3.5 for TB4, 1.0 for 10GbE, 0.2 for Wi-Fi)
    #[arg(long, default_value_t = 1.0)]
    pub bandwidth_gbps: f64,

    /// Measured round-trip latency in milliseconds
    #[arg(long, default_value_t = 0.25)]
    pub rtt_ms: f64,

    /// Total transformer layers to partition across nodes
    #[arg(long, default_value_t = 32)]
    pub layers: usize,

    /// Node specs as `id:vram_gb:avg_bpw:mem_bw_gbps` (repeatable; defaults to 2-node example)
    #[arg(long = "node")]
    pub nodes: Vec<String>,

    /// Output JSON instead of text
    #[arg(long)]
    pub json: bool,
}

fn parse_node_spec(spec: &str) -> Result<ClusterNode> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 4 {
        bail!("invalid --node '{spec}': expected id:vram_gb:avg_bpw:mem_bw_gbps");
    }
    let vram_gb: f64 = parts[1]
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid vram_gb in '{spec}': {e}"))?;
    let avg_bpw: f64 = parts[2]
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid avg_bpw in '{spec}': {e}"))?;
    let mem_bandwidth_gbps: f64 = parts[3]
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid mem_bw_gbps in '{spec}': {e}"))?;
    Ok(ClusterNode {
        id: parts[0].to_string(),
        vram_avail_bytes: (vram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
        avg_bpw,
        mem_bandwidth_gbps,
    })
}

fn wire_codec_label(mode: ClusterParallelMode) -> &'static str {
    match mode {
        ClusterParallelMode::TensorAndPipeline => "Q8_1 activations + BITCOS shard streaming",
        ClusterParallelMode::CoarsePipelineQ41 => {
            "Hadamard Q4_1 stage-boundary wire (0.5625 B/elem)"
        }
        ClusterParallelMode::AsyncEdgeDraftVerify => "32B token IDs + QJL page IDs (zero sync TP)",
    }
}

pub fn execute(args: ClusterArgs) -> Result<()> {
    let (tier, mode) = classify_interconnect(args.bandwidth_gbps, args.rtt_ms);
    let nodes = if args.nodes.is_empty() {
        vec![
            ClusterNode {
                id: "primary-gpu".to_string(),
                vram_avail_bytes: 16 * 1024 * 1024 * 1024,
                avg_bpw: 4.5,
                mem_bandwidth_gbps: 288.0,
            },
            ClusterNode {
                id: "edge-bitcos".to_string(),
                vram_avail_bytes: 8 * 1024 * 1024 * 1024,
                avg_bpw: 1.610,
                mem_bandwidth_gbps: 90.0,
            },
        ]
    } else {
        args.nodes
            .iter()
            .map(|s| parse_node_spec(s))
            .collect::<Result<Vec<_>>>()?
    };

    let stages = partition_stages(&nodes, args.layers)?;
    let wire = wire_codec_label(mode);
    if args.json {
        let out = serde_json::json!({
            "interconnect_tier": format!("{tier:?}"),
            "parallel_mode": format!("{mode:?}"),
            "wire_codec": wire,
            "stages": stages.iter().map(|s| serde_json::json!({
                "node_id": s.node_id,
                "start_layer": s.start_layer,
                "end_layer": s.end_layer,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("THME Heterogeneous Cluster Planner (§7.12)");
        println!(
            "Interconnect: {tier:?} ({:.2} GB/s, {:.2} ms RTT)",
            args.bandwidth_gbps, args.rtt_ms
        );
        println!("Mode: {mode:?} | Wire: {wire}");
        for s in &stages {
            println!(
                "  - {} -> layers [{}..{})",
                s.node_id, s.start_layer, s.end_layer
            );
        }
    }
    Ok(())
}
