// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PLM_MASK_LEVEL_SHIFT: u32 = 0;
pub const PLM_LEVEL_ENABLE: u32 = 0x0000_0001;
pub const PLM_UNLOCK_CLEAR: u32 = 0x0000_0000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterChange {
    pub offset: u32,
    pub register_name: String,
    pub original_value: u32,
    pub new_value: u32,
}

/// Abstract representation of BAR0 MMIO space.
/// Safe simulation memory space with optional live hardware pointer.
// shortcut: simulated HashMap MMIO aperture with safe dry-run, upgrade to /dev/vfio/ or mmap on resource0 when live hardware access is explicitly granted
#[derive(Debug)]
pub struct Bar0Aperture {
    simulated_memory: HashMap<u32, u32>,
    dry_run: bool,
    history: Vec<RegisterChange>,
}

impl Bar0Aperture {
    pub fn new_simulated(dry_run: bool) -> Self {
        Self {
            simulated_memory: HashMap::new(),
            dry_run,
            history: Vec::new(),
        }
    }

    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    pub fn read_u32(&self, offset: u32) -> u32 {
        *self.simulated_memory.get(&offset).unwrap_or(&0)
    }

    pub fn write_u32(&mut self, offset: u32, value: u32, name: &str) {
        let original = self.read_u32(offset);
        self.history.push(RegisterChange {
            offset,
            register_name: name.to_string(),
            original_value: original,
            new_value: value,
        });

        if !self.dry_run {
            self.simulated_memory.insert(offset, value);
        }
    }

    pub fn history(&self) -> &[RegisterChange] {
        &self.history
    }
}
