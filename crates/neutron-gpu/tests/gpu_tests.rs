// SPDX-License-Identifier: AGPL-3.0-only
use neutron_gpu::pci::PciScanner;
use neutron_gpu::recipes::{all_recipes, find_recipe, ChipFamily};

#[test]
fn test_recipe_lookup_and_families() {
    let turing = find_recipe(0x1e04).expect("RTX 2080 Ti must exist in recipes");
    assert_eq!(turing.family, ChipFamily::Turing);
    assert_eq!(turing.chip_id, 0x162);

    let ampere = find_recipe(0x2204).expect("RTX 3090 must exist in recipes");
    assert_eq!(ampere.family, ChipFamily::Ampere);
    assert_eq!(ampere.chip_id, 0x172);

    let ada = find_recipe(0x2684).expect("RTX 4090 must exist in recipes");
    assert_eq!(ada.family, ChipFamily::AdaLovelace);
    assert_eq!(ada.chip_id, 0x192);

    // Legacy Architecture Checks (Pascal & Maxwell)
    let pascal = find_recipe(0x1b06).expect("GTX 1080 Ti must exist in recipes");
    assert_eq!(pascal.family, ChipFamily::Pascal);
    assert!(!pascal.has_tensor_cores());

    let maxwell = find_recipe(0x13c0).expect("GTX 980 must exist in recipes");
    assert_eq!(maxwell.family, ChipFamily::Maxwell);
    assert!(!maxwell.has_tensor_cores());

    assert!(all_recipes().len() >= 40, "Database must hold 40+ recipes");
}

#[test]
fn test_rtx4060ti_vram_recipe_refinement() {
    use neutron_gpu::recipes::find_recipe_with_vram;

    // Default (no VRAM info) resolves to the 8GB variant
    let default_recipe = find_recipe_with_vram(0x2803, None).expect("0x2803 recipe must exist");
    assert_eq!(default_recipe.name, "GeForce RTX 4060 Ti (8GB)");

    // 8GB detected VRAM keeps the 8GB variant
    let recipe_8gb = find_recipe_with_vram(0x2803, Some(8 * 1024 * 1024 * 1024))
        .expect("0x2803 8GB recipe must exist");
    assert_eq!(recipe_8gb.name, "GeForce RTX 4060 Ti (8GB)");

    // 16GB detected VRAM resolves to the 16GB variant
    let recipe_16gb = find_recipe_with_vram(0x2803, Some(16 * 1024 * 1024 * 1024))
        .expect("0x2803 16GB recipe must exist");
    assert_eq!(recipe_16gb.name, "GeForce RTX 4060 Ti (16GB)");
    assert_eq!(recipe_16gb.chip_id, 0x196);
    assert_eq!(recipe_16gb.family, ChipFamily::AdaLovelace);
}

#[test]
fn test_pci_scanner_fallback() {
    let devs = PciScanner::scan_fallback();
    assert!(
        devs.len() >= 5,
        "Fallback scan must return multi-tier accelerators"
    );
    assert_eq!(devs[0].device_id, 0x2204);
    assert!(devs[0].matched_recipe.is_some());

    // Verify presence of APU, NPU, Apple Silicon, and TPU tiers
    assert!(devs
        .iter()
        .any(|d| d.tier == neutron_gpu::pci::DeviceTier::IntegratedApu));
    assert!(devs
        .iter()
        .any(|d| d.tier == neutron_gpu::pci::DeviceTier::NeuralNpu));
    assert!(devs
        .iter()
        .any(|d| d.tier == neutron_gpu::pci::DeviceTier::ArmSilicon));
    assert!(devs
        .iter()
        .any(|d| d.tier == neutron_gpu::pci::DeviceTier::TensorTpu));
}
