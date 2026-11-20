// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChipFamily {
    Maxwell,
    Pascal,
    Volta,
    Turing,
    Ampere,
    AdaLovelace,
}

impl ChipFamily {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Maxwell => "Maxwell",
            Self::Pascal => "Pascal",
            Self::Volta => "Volta",
            Self::Turing => "Turing",
            Self::Ampere => "Ampere",
            Self::AdaLovelace => "Ada Lovelace",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CardRecipe {
    pub device_id: u16,
    pub name: &'static str,
    pub family: ChipFamily,
    pub chip_id: u16,
}

impl CardRecipe {
    pub fn has_tensor_cores(&self) -> bool {
        matches!(
            self.family,
            ChipFamily::Volta | ChipFamily::Turing | ChipFamily::Ampere | ChipFamily::AdaLovelace
        )
    }
}

pub static CARD_RECIPES: &[CardRecipe] = &[
    // --- Maxwell (GM200 / GM204 / GM206) - Legacy Architecture Support ---
    CardRecipe {
        device_id: 0x17c2,
        name: "GeForce GTX TITAN X (Maxwell GM200)",
        family: ChipFamily::Maxwell,
        chip_id: 0x120,
    },
    CardRecipe {
        device_id: 0x17c8,
        name: "GeForce GTX 980 Ti (GM200)",
        family: ChipFamily::Maxwell,
        chip_id: 0x120,
    },
    CardRecipe {
        device_id: 0x13c0,
        name: "GeForce GTX 980 (GM204)",
        family: ChipFamily::Maxwell,
        chip_id: 0x124,
    },
    CardRecipe {
        device_id: 0x13c2,
        name: "GeForce GTX 970 (GM204)",
        family: ChipFamily::Maxwell,
        chip_id: 0x124,
    },
    CardRecipe {
        device_id: 0x1401,
        name: "GeForce GTX 960 (GM206)",
        family: ChipFamily::Maxwell,
        chip_id: 0x126,
    },
    CardRecipe {
        device_id: 0x1402,
        name: "GeForce GTX 950 (GM206)",
        family: ChipFamily::Maxwell,
        chip_id: 0x126,
    },
    // --- Pascal (GP100 / GP102 / GP104 / GP106 / GP107) - Legacy Architecture Support ---
    CardRecipe {
        device_id: 0x1b00,
        name: "TITAN X (Pascal GP102)",
        family: ChipFamily::Pascal,
        chip_id: 0x132,
    },
    CardRecipe {
        device_id: 0x1b02,
        name: "TITAN Xp (GP102)",
        family: ChipFamily::Pascal,
        chip_id: 0x132,
    },
    CardRecipe {
        device_id: 0x1b06,
        name: "GeForce GTX 1080 Ti (11GB)",
        family: ChipFamily::Pascal,
        chip_id: 0x132,
    },
    CardRecipe {
        device_id: 0x1b80,
        name: "GeForce GTX 1080 (8GB GP104)",
        family: ChipFamily::Pascal,
        chip_id: 0x134,
    },
    CardRecipe {
        device_id: 0x1b81,
        name: "GeForce GTX 1070 (8GB)",
        family: ChipFamily::Pascal,
        chip_id: 0x134,
    },
    CardRecipe {
        device_id: 0x1b82,
        name: "GeForce GTX 1070 Ti",
        family: ChipFamily::Pascal,
        chip_id: 0x134,
    },
    CardRecipe {
        device_id: 0x1c03,
        name: "GeForce GTX 1060 6GB (GP106)",
        family: ChipFamily::Pascal,
        chip_id: 0x136,
    },
    CardRecipe {
        device_id: 0x1c02,
        name: "GeForce GTX 1060 3GB",
        family: ChipFamily::Pascal,
        chip_id: 0x136,
    },
    CardRecipe {
        device_id: 0x1c82,
        name: "GeForce GTX 1050 Ti (GP107)",
        family: ChipFamily::Pascal,
        chip_id: 0x137,
    },
    CardRecipe {
        device_id: 0x1c81,
        name: "GeForce GTX 1050",
        family: ChipFamily::Pascal,
        chip_id: 0x137,
    },
    // --- Volta (GV100) ---
    CardRecipe {
        device_id: 0x1d81,
        name: "TITAN V (GV100)",
        family: ChipFamily::Volta,
        chip_id: 0x140,
    },
    // --- Turing (TU102 / TU104 / TU106) ---
    CardRecipe {
        device_id: 0x1e03,
        name: "GeForce RTX 2080 Ti (TU102 Core)",
        family: ChipFamily::Turing,
        chip_id: 0x162,
    },
    CardRecipe {
        device_id: 0x1e04,
        name: "GeForce RTX 2080 Ti",
        family: ChipFamily::Turing,
        chip_id: 0x162,
    },
    CardRecipe {
        device_id: 0x1e07,
        name: "GeForce RTX 2080 Ti Rev. A",
        family: ChipFamily::Turing,
        chip_id: 0x162,
    },
    CardRecipe {
        device_id: 0x1e2e,
        name: "NVIDIA TITAN RTX",
        family: ChipFamily::Turing,
        chip_id: 0x162,
    },
    CardRecipe {
        device_id: 0x1e81,
        name: "GeForce RTX 2080 (TU104 Core)",
        family: ChipFamily::Turing,
        chip_id: 0x164,
    },
    CardRecipe {
        device_id: 0x1e82,
        name: "GeForce RTX 2080 Super",
        family: ChipFamily::Turing,
        chip_id: 0x164,
    },
    CardRecipe {
        device_id: 0x1e84,
        name: "GeForce RTX 2070 Super",
        family: ChipFamily::Turing,
        chip_id: 0x164,
    },
    CardRecipe {
        device_id: 0x1e87,
        name: "GeForce RTX 2080 (OEM)",
        family: ChipFamily::Turing,
        chip_id: 0x164,
    },
    CardRecipe {
        device_id: 0x1f02,
        name: "GeForce RTX 2070 (TU106)",
        family: ChipFamily::Turing,
        chip_id: 0x166,
    },
    CardRecipe {
        device_id: 0x1f03,
        name: "GeForce RTX 2060 Super",
        family: ChipFamily::Turing,
        chip_id: 0x166,
    },
    CardRecipe {
        device_id: 0x1f06,
        name: "GeForce RTX 2060 6GB",
        family: ChipFamily::Turing,
        chip_id: 0x166,
    },
    CardRecipe {
        device_id: 0x1f07,
        name: "GeForce RTX 2060 12GB",
        family: ChipFamily::Turing,
        chip_id: 0x166,
    },
    // --- Ampere (GA100 / GA102 / GA103 / GA104 / GA106) ---
    CardRecipe {
        device_id: 0x2200,
        name: "GeForce RTX 3090 (GA102 Prototype)",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2203,
        name: "GeForce RTX 3090 Ti",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2204,
        name: "GeForce RTX 3090",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2205,
        name: "GeForce RTX 3080 Ti (12GB)",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2206,
        name: "GeForce RTX 3080 (10GB)",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2207,
        name: "GeForce RTX 3080 (OEM)",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2208,
        name: "GeForce RTX 3080 Ti (12GB GA102)",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2216,
        name: "GeForce RTX 3080 (12GB LHR)",
        family: ChipFamily::Ampere,
        chip_id: 0x172,
    },
    CardRecipe {
        device_id: 0x2414,
        name: "GeForce RTX 3080 Ti Laptop (GA103)",
        family: ChipFamily::Ampere,
        chip_id: 0x173,
    },
    CardRecipe {
        device_id: 0x2480,
        name: "GeForce RTX 3070 Ti (GA104)",
        family: ChipFamily::Ampere,
        chip_id: 0x174,
    },
    CardRecipe {
        device_id: 0x2482,
        name: "GeForce RTX 3070 Ti (GA104-400)",
        family: ChipFamily::Ampere,
        chip_id: 0x174,
    },
    CardRecipe {
        device_id: 0x2484,
        name: "GeForce RTX 3070 (GA104)",
        family: ChipFamily::Ampere,
        chip_id: 0x174,
    },
    CardRecipe {
        device_id: 0x2486,
        name: "GeForce RTX 3060 Ti (GA104)",
        family: ChipFamily::Ampere,
        chip_id: 0x174,
    },
    CardRecipe {
        device_id: 0x2487,
        name: "GeForce RTX 3060 Ti GDDR6X",
        family: ChipFamily::Ampere,
        chip_id: 0x174,
    },
    CardRecipe {
        device_id: 0x2501,
        name: "GeForce RTX 3060 (GA106 Core)",
        family: ChipFamily::Ampere,
        chip_id: 0x176,
    },
    CardRecipe {
        device_id: 0x2503,
        name: "GeForce RTX 3060 12GB",
        family: ChipFamily::Ampere,
        chip_id: 0x176,
    },
    CardRecipe {
        device_id: 0x2504,
        name: "GeForce RTX 3060 8GB",
        family: ChipFamily::Ampere,
        chip_id: 0x176,
    },
    // --- Ada Lovelace (AD102 / AD103 / AD104 / AD106 / AD107) ---
    CardRecipe {
        device_id: 0x2684,
        name: "GeForce RTX 4090 (24GB)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x192,
    },
    CardRecipe {
        device_id: 0x2685,
        name: "GeForce RTX 4090 D",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x192,
    },
    CardRecipe {
        device_id: 0x2702,
        name: "GeForce RTX 4080 (16GB)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x193,
    },
    CardRecipe {
        device_id: 0x2704,
        name: "GeForce RTX 4080 Super",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x193,
    },
    CardRecipe {
        device_id: 0x2705,
        name: "GeForce RTX 4080 (OEM)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x193,
    },
    CardRecipe {
        device_id: 0x2757,
        name: "GeForce RTX 4090 Laptop (AD103)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x193,
    },
    CardRecipe {
        device_id: 0x2782,
        name: "GeForce RTX 4070 Ti (AD104)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x194,
    },
    CardRecipe {
        device_id: 0x2783,
        name: "GeForce RTX 4070 Ti Super (AD103/104)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x194,
    },
    CardRecipe {
        device_id: 0x2786,
        name: "GeForce RTX 4070 / Super",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x194,
    },
    CardRecipe {
        device_id: 0x2803,
        name: "GeForce RTX 4060 Ti (8GB)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x196,
    },
    CardRecipe {
        device_id: 0x2805,
        name: "GeForce RTX 4060 Ti (16GB)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x196,
    },
    CardRecipe {
        device_id: 0x2882,
        name: "GeForce RTX 4060 (AD107)",
        family: ChipFamily::AdaLovelace,
        chip_id: 0x197,
    },
];

pub fn find_recipe(device_id: u16) -> Option<&'static CardRecipe> {
    CARD_RECIPES.iter().find(|r| r.device_id == device_id)
}

/// Finds a card recipe, refining the model naming if detected VRAM capacity indicates a distinct variant.
/// Specifically, NVIDIA GeForce RTX 4060 Ti desktop cards report device ID 0x2803 for both 8GB and 16GB
/// retail models; detected VRAM >= 12 GiB resolves to the 16GB variant.
pub fn find_recipe_with_vram(device_id: u16, vram_bytes: Option<u64>) -> Option<CardRecipe> {
    let mut recipe = find_recipe(device_id).cloned()?;
    if device_id == 0x2803 {
        if let Some(bytes) = vram_bytes {
            if bytes >= 12 * 1024 * 1024 * 1024 {
                recipe.name = "GeForce RTX 4060 Ti (16GB)";
            }
        }
    }
    Some(recipe)
}

pub fn get_recipe(vendor_id: u16, device_id: u16) -> Option<&'static CardRecipe> {
    if vendor_id == crate::pci::PCI_VENDOR_NVIDIA {
        find_recipe(device_id)
    } else {
        None
    }
}

/// Resolves a card recipe by vendor and device ID, refining by VRAM capacity if available.
pub fn get_recipe_with_vram(
    vendor_id: u16,
    device_id: u16,
    vram_bytes: Option<u64>,
) -> Option<CardRecipe> {
    if vendor_id == crate::pci::PCI_VENDOR_NVIDIA {
        find_recipe_with_vram(device_id, vram_bytes)
    } else {
        None
    }
}

pub fn all_recipes() -> &'static [CardRecipe] {
    CARD_RECIPES
}
