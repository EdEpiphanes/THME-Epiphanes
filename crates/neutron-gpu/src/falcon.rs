// SPDX-License-Identifier: AGPL-3.0-only
use crate::recipes::{CardRecipe, ChipFamily};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RopStackFrame {
    pub return_address: u32,
    pub target_register: u32,
    pub write_value: u32,
    pub stack_padding: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FalconPayload {
    pub family: ChipFamily,
    pub chip_id: u16,
    pub dmem_offset: u32,
    pub frames: Vec<RopStackFrame>,
    pub total_bytes: usize,
}

pub struct FalconRopSynthesizer;

impl FalconRopSynthesizer {
    /// Constructs a Falcon SEC2 High Secure mode ROP stack frame
    /// to clear the PLM write protect bitmasks for the specified GPU recipe.
    pub fn synthesize(recipe: &CardRecipe) -> FalconPayload {
        let (dmem_base, plm_clear_value) = match recipe.family {
            ChipFamily::Maxwell | ChipFamily::Pascal => (0x0000_0400, 0x0000_0000),
            ChipFamily::Volta | ChipFamily::Turing => (0x0000_0800, 0x0000_0000),
            ChipFamily::Ampere => (0x0000_1000, 0x0000_0000),
            ChipFamily::AdaLovelace => (0x0000_1800, 0x0000_0000),
        };

        // V67 ROP chain: Overwrites stack frame return address to point to indirect register write gadget
        let frames = vec![RopStackFrame {
            return_address: recipe.rop_gadget,
            target_register: recipe.plm_feat_offset,
            write_value: plm_clear_value,
            stack_padding: vec![0xDEAD_BEEF, 0x0000_0000],
        }];

        let total_bytes = frames.len() * (4 * 4); // return_addr + target + write_val + padding

        FalconPayload {
            family: recipe.family,
            chip_id: recipe.chip_id,
            dmem_offset: dmem_base,
            frames,
            total_bytes,
        }
    }
}
