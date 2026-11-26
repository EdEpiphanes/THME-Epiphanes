// SPDX-License-Identifier: AGPL-3.0-only
//! §7.8 Smart Compressed Off-Loading (VRAM / RAM / CPU / SSD) & Safety Guardrails.

use anyhow::{bail, Result};

/// Maximum prefill tokens per WDDM dispatch to prevent TDR timeouts (< 1.5 s/dispatch).
pub const WDDM_MAX_PREFILL_CHUNK: usize = 512;
/// 4 KiB sector alignment for direct NVMe I/O (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`).
pub const NVME_SECTOR_ALIGN_BYTES: usize = 4096;
/// Maximum pinned host staging ring capacity (4 GiB).
pub const MAX_PINNED_HOST_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// 4-tier storage placement for weights and KV pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadTier {
    VramResident,
    PinnedHostRam,
    CpuColdKvQjl,
    NvmeSectorArena,
}

/// Online EWMA tracker for measured PCIe Gen3/4/5 bandwidth (`bytes/sec`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcieBandwidthEwma {
    pub bytes_per_sec: f64,
    pub alpha: f64,
}

impl Default for PcieBandwidthEwma {
    fn default() -> Self {
        Self {
            bytes_per_sec: 24.0e9,
            alpha: 0.2,
        }
    }
}

impl PcieBandwidthEwma {
    /// Updates the EWMA bandwidth estimate from an observed DMA transfer.
    pub fn observe(&mut self, bytes: u64, elapsed_secs: f64) {
        if bytes == 0 || elapsed_secs <= 0.0 {
            return;
        }
        let sample = bytes as f64 / elapsed_secs;
        self.bytes_per_sec = self.alpha * sample + (1.0 - self.alpha) * self.bytes_per_sec;
    }
}

/// Co-located block descriptor for Residency Utility ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct CoLocatedBlock {
    pub id: String,
    pub is_boundary_or_router: bool,
    pub bytes: u64,
    pub reuse_freq: f64,
    pub overlap_compute_secs: f64,
}

/// Computes `U_res(G) = (f_reuse(G) * max(0, bytes(G)/B_PCIe - t_compute)) / bytes(G)`.
#[must_use]
pub fn residency_utility(block: &CoLocatedBlock, pcie_bytes_per_sec: f64) -> f64 {
    if block.is_boundary_or_router {
        return f64::INFINITY;
    }
    if block.bytes == 0 || pcie_bytes_per_sec <= 0.0 {
        return 0.0;
    }
    let dma_secs = block.bytes as f64 / pcie_bytes_per_sec;
    let stall_secs = (dma_secs - block.overlap_compute_secs.max(0.0)).max(0.0);
    (block.reuse_freq.max(0.0) * stall_secs) / block.bytes as f64
}

/// Computes the safe VRAM budget (`0.85 * V_free`) and pinned host RAM cap (`min(4 GiB, 0.25 * host_ram_free)`).
#[must_use]
pub fn compute_memory_budgets(vram_free_bytes: u64, host_ram_free_bytes: u64) -> (u64, u64) {
    let vram_avail = ((vram_free_bytes as f64) * 0.85) as u64;
    let host_cap = (host_ram_free_bytes / 4).min(MAX_PINNED_HOST_BYTES);
    (vram_avail, host_cap)
}

/// Assigns co-located blocks to `VramResident` up to `vram_avail_bytes`, spilling
/// remaining non-boundary blocks intact to `PinnedHostRam` or `NvmeSectorArena`.
pub fn plan_block_residency(
    blocks: &[CoLocatedBlock],
    vram_avail_bytes: u64,
    pinned_host_cap_bytes: u64,
    pcie_bytes_per_sec: f64,
) -> Vec<(String, OffloadTier)> {
    let mut ranked: Vec<(usize, f64)> = blocks
        .iter()
        .enumerate()
        .map(|(idx, b)| (idx, residency_utility(b, pcie_bytes_per_sec)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut placements = vec![OffloadTier::NvmeSectorArena; blocks.len()];
    let mut vram_used = 0u64;
    let mut host_used = 0u64;

    for (idx, _) in ranked {
        let block = &blocks[idx];
        if block.is_boundary_or_router || vram_used.saturating_add(block.bytes) <= vram_avail_bytes
        {
            vram_used = vram_used.saturating_add(block.bytes);
            placements[idx] = OffloadTier::VramResident;
        } else if host_used.saturating_add(block.bytes) <= pinned_host_cap_bytes {
            host_used = host_used.saturating_add(block.bytes);
            placements[idx] = OffloadTier::PinnedHostRam;
        } else {
            placements[idx] = OffloadTier::NvmeSectorArena;
        }
    }

    blocks
        .iter()
        .zip(placements)
        .map(|(b, tier)| (b.id.clone(), tier))
        .collect()
}

/// Computes adaptive speculative horizon `K = min(K_VRAM, floor((1 - alpha_spec)^-1))` in `1..=8`.
#[must_use]
pub fn adaptive_speculative_horizon(k_vram: usize, alpha_spec: f32) -> usize {
    let denom = (1.0 - alpha_spec.clamp(0.0, 0.95)).max(0.05);
    let k_alpha = (1.0 / denom).floor() as usize;
    k_vram.min(k_alpha).clamp(1, 8)
}

/// Chunks a prefill sequence length into `<= 512`-token WDDM-safe windows.
#[must_use]
pub fn wddm_prefill_chunks(total_tokens: usize) -> Vec<usize> {
    if total_tokens == 0 {
        return Vec::new();
    }
    let full = total_tokens / WDDM_MAX_PREFILL_CHUNK;
    let rem = total_tokens % WDDM_MAX_PREFILL_CHUNK;
    let mut chunks = vec![WDDM_MAX_PREFILL_CHUNK; full];
    if rem > 0 {
        chunks.push(rem);
    }
    chunks
}

/// Bounds-clamped `BITCOS` super-block index lookup so bit-flips cannot cause OOB reads.
pub fn clamp_bitcos_superblock_idx(raw_idx: usize, superblock_nz_cap: usize) -> Result<usize> {
    if superblock_nz_cap == 0 {
        bail!("superblock_nz_cap must be positive");
    }
    Ok(raw_idx.min(superblock_nz_cap - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_blocks_always_vram_resident_and_host_capped() {
        let blocks = vec![
            CoLocatedBlock {
                id: "lm_head".to_string(),
                is_boundary_or_router: true,
                bytes: 200,
                reuse_freq: 1.0,
                overlap_compute_secs: 0.0,
            },
            CoLocatedBlock {
                id: "mlp_1".to_string(),
                is_boundary_or_router: false,
                bytes: 200,
                reuse_freq: 2.0,
                overlap_compute_secs: 0.0,
            },
            CoLocatedBlock {
                id: "mlp_2".to_string(),
                is_boundary_or_router: false,
                bytes: 200,
                reuse_freq: 0.5,
                overlap_compute_secs: 0.0,
            },
        ];
        let plan = plan_block_residency(&blocks, 350, 250, 24.0e9);
        assert_eq!(plan[0].1, OffloadTier::VramResident);
        assert_eq!(plan[1].1, OffloadTier::PinnedHostRam);
        assert_eq!(plan[2].1, OffloadTier::NvmeSectorArena);
    }

    #[test]
    fn speculative_horizon_wddm_chunks_and_bitcos_clamp_work() -> Result<()> {
        assert_eq!(adaptive_speculative_horizon(8, 0.75), 4);
        assert_eq!(adaptive_speculative_horizon(2, 0.90), 2);
        assert_eq!(wddm_prefill_chunks(1025), vec![512, 512, 1]);
        assert_eq!(clamp_bitcos_superblock_idx(9999, 64)?, 63);
        Ok(())
    }
}
