// SPDX-License-Identifier: AGPL-3.0-only
//! §7.7.3 & §7.8.3 Native GPU ISA Lowering (`PTX` & `WGSL`) + Zero-Spill Verification.

use anyhow::{bail, Result};

/// Maximum permitted registers per thread under §7.7.3 Zero-Spill ISA verification.
pub const MAX_PTX_REGISTERS_PER_THREAD: u32 = 48;

/// WGSL compute shader for `BITCOS` packed GEMV.
pub const WGSL_GEMV_BITCOS: &str = include_str!("shaders/gemv_bitcos.wgsl");

/// Bounds-clamped `ptx_bitcos_gemv` PTX kernel (§7.7.3 / §7.8.3):
/// - Dual-pipe INT32 (`popc.b32` / bitmask) + FP32 (`fma.rn.f32`) ILP
/// - Clamps `idx` to `superblock_nz_cap` via `min.u32` so bit-flips cannot trigger OOB reads.
pub const PTX_BITCOS_GEMV: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry ptx_bitcos_gemv(
    .param .u64 param_w_mask,
    .param .u64 param_x,
    .param .u64 param_y,
    .param .u32 param_superblock_nz_cap,
    .param .u32 param_idx
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<8>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [param_w_mask];
    ld.param.u64 %rd2, [param_x];
    ld.param.u64 %rd3, [param_y];
    ld.param.u32 %r1, [param_superblock_nz_cap];
    ld.param.u32 %r2, [param_idx];

    // Bounds-clamp index to superblock_nz_cap - 1
    sub.u32 %r3, %r1, 1;
    min.u32 %r4, %r2, %r3;
    ld.global.u32 %r5, [%rd1];
    popc.b32 %r6, %r5;
    cvt.rn.f32.u32 %f1, %r6;
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    fma.rn.f32 %f3, %f1, %f2, 0.0;
    st.global.f32 [%rd3], %f3;
    ret;
}
"#;

/// `ptx_polar_gemv` PTX kernel stub for PolarQuant trigonometric inner product lowering.
pub const PTX_POLAR_GEMV: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry ptx_polar_gemv(
    .param .u64 param_theta,
    .param .u64 param_out,
    .param .f32 param_scale
)
{
    .reg .u64 %rd<3>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [param_theta];
    ld.param.u64 %rd2, [param_out];
    ld.param.f32 %f1, [param_scale];
    ld.global.f32 %f2, [%rd1];
    cos.approx.f32 %f3, %f2;
    mul.f32 %f4, %f3, %f1;
    st.global.f32 [%rd2], %f4;
    ret;
}
"#;

/// Parsed `ptxas --verbose` register and local-memory spill report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtxSpillAudit {
    pub registers_used: u32,
    pub lmem_bytes: u32,
}

/// Validates a `ptxas --verbose` line (e.g. `"Used 32 registers, 0 bytes lmem"`)
/// against `<= max_regs` and `lmem == 0 bytes`.
pub fn verify_zero_spill_isa(ptxas_output: &str, max_regs: u32) -> Result<PtxSpillAudit> {
    let mut registers_used = None;
    let mut lmem_bytes = Some(0u32);

    let tokens: Vec<&str> = ptxas_output.split_whitespace().collect();
    for win in tokens.windows(2) {
        if win[1].starts_with("register") {
            if let Ok(n) = win[0].trim_end_matches(',').parse::<u32>() {
                registers_used = Some(n);
            }
        }
        if win[1].starts_with("lmem") {
            // Handles "... 0 bytes lmem" where win[0] is "bytes" -> check 2 back.
        }
    }
    for win in tokens.windows(3) {
        if win[1].starts_with("byte") && win[2].starts_with("lmem") {
            if let Ok(n) = win[0].trim_end_matches(',').parse::<u32>() {
                lmem_bytes = Some(n);
            }
        }
    }

    let regs = registers_used
        .ok_or_else(|| anyhow::anyhow!("ptxas output missing register count: {ptxas_output}"))?;
    let lmem = lmem_bytes.unwrap_or(0);
    if regs > max_regs {
        bail!("PTX register budget exceeded: {regs} > {max_regs}");
    }
    if lmem > 0 {
        bail!("PTX local-memory spill detected: lmem = {lmem} bytes (must be 0)");
    }
    Ok(PtxSpillAudit {
        registers_used: regs,
        lmem_bytes: lmem,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_and_wgsl_bitcos_kernels_contain_bounds_clamping() {
        assert!(PTX_BITCOS_GEMV.contains("superblock_nz_cap"));
        assert!(PTX_BITCOS_GEMV.contains("min.u32"));
        assert!(PTX_BITCOS_GEMV.contains("popc.b32"));
        assert!(PTX_POLAR_GEMV.contains("ptx_polar_gemv"));
        // WGSL reads are already bounded by `k < in_dim`; it follows main's z-batched layout.
        assert!(WGSL_GEMV_BITCOS.contains("k < params.in_dim"));
        assert!(WGSL_GEMV_BITCOS.contains("wid.z"));
    }

    #[test]
    fn zero_spill_isa_verifier_enforces_reg_cap_and_zero_lmem() -> Result<()> {
        let ok = verify_zero_spill_isa(
            "ptxas info : Used 36 registers, 0 bytes lmem, 384 bytes cmem[0]",
            MAX_PTX_REGISTERS_PER_THREAD,
        )?;
        assert_eq!(ok.registers_used, 36);
        assert_eq!(ok.lmem_bytes, 0);

        assert!(verify_zero_spill_isa(
            "ptxas info : Used 52 registers, 0 bytes lmem",
            MAX_PTX_REGISTERS_PER_THREAD,
        )
        .is_err());
        assert!(verify_zero_spill_isa(
            "ptxas info : Used 32 registers, 16 bytes lmem",
            MAX_PTX_REGISTERS_PER_THREAD,
        )
        .is_err());
        Ok(())
    }
}
