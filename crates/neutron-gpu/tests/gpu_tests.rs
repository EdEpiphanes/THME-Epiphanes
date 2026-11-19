use neutron_gpu::engine::UnlockEngine;
use neutron_gpu::falcon::FalconRopSynthesizer;
use neutron_gpu::mdev::MdevManager;
use neutron_gpu::pci::{PciDevice, PciScanner};
use neutron_gpu::recipes::{
    all_recipes, find_recipe, ChipFamily, ADA_PLM_FEAT, AMPERE_PLM_FEAT, TURING_PLM_FEAT,
    UNLOCK_RATE_FULL_IMLA4, UNLOCK_RATE_FULL_MMA,
};

#[test]
fn test_recipe_lookup_and_families() {
    let turing = find_recipe(0x1e04).expect("RTX 2080 Ti must exist in recipes");
    assert_eq!(turing.family, ChipFamily::Turing);
    assert_eq!(turing.plm_feat_offset, TURING_PLM_FEAT);
    assert_eq!(turing.chip_id, 0x162);

    let ampere = find_recipe(0x2204).expect("RTX 3090 must exist in recipes");
    assert_eq!(ampere.family, ChipFamily::Ampere);
    assert_eq!(ampere.plm_feat_offset, AMPERE_PLM_FEAT);
    assert_eq!(ampere.chip_id, 0x172);

    let ada = find_recipe(0x2684).expect("RTX 4090 must exist in recipes");
    assert_eq!(ada.family, ChipFamily::AdaLovelace);
    assert_eq!(ada.plm_feat_offset, ADA_PLM_FEAT);
    assert_eq!(ada.chip_id, 0x192);

    // Legacy Architecture Checks (Pascal & Maxwell)
    let pascal = find_recipe(0x1b06).expect("GTX 1080 Ti must exist in recipes");
    assert_eq!(pascal.family, ChipFamily::Pascal);
    assert!(!pascal.has_tensor_cores());
    assert!(pascal.has_vgpu_support());

    let maxwell = find_recipe(0x13c0).expect("GTX 980 must exist in recipes");
    assert_eq!(maxwell.family, ChipFamily::Maxwell);
    assert!(!maxwell.has_tensor_cores());
    assert!(maxwell.has_vgpu_support());

    assert!(all_recipes().len() >= 40, "Database must hold 40+ recipes");
}

#[test]
fn test_rtx4060ti_vram_recipe_refinement() {
    use neutron_gpu::recipes::find_recipe_with_vram;

    // Default lookup without VRAM
    let default_recipe = find_recipe_with_vram(0x2803, None).expect("0x2803 recipe must exist");
    assert_eq!(default_recipe.name, "GeForce RTX 4060 Ti (8GB)");

    // 8GB card lookup
    let recipe_8gb = find_recipe_with_vram(0x2803, Some(8 * 1024 * 1024 * 1024))
        .expect("0x2803 8GB recipe must exist");
    assert_eq!(recipe_8gb.name, "GeForce RTX 4060 Ti (8GB)");

    // 16GB card lookup refined by capacity
    let recipe_16gb = find_recipe_with_vram(0x2803, Some(16 * 1024 * 1024 * 1024))
        .expect("0x2803 16GB recipe must exist");
    assert_eq!(recipe_16gb.name, "GeForce RTX 4060 Ti (16GB)");
    assert_eq!(recipe_16gb.chip_id, 0x196);
    assert_eq!(recipe_16gb.family, ChipFamily::AdaLovelace);
}

#[test]
fn test_falcon_rop_synthesizer() {
    let ampere_recipe = find_recipe(0x2204).unwrap();
    let payload = FalconRopSynthesizer::synthesize(ampere_recipe);

    assert_eq!(payload.family, ChipFamily::Ampere);
    assert_eq!(payload.chip_id, 0x172);
    assert_eq!(payload.frames.len(), 1);

    let frame = &payload.frames[0];
    assert_eq!(frame.return_address, ampere_recipe.rop_gadget);
    assert_eq!(frame.target_register, ampere_recipe.plm_feat_offset);
    assert_eq!(frame.write_value, 0x0000_0000); // clears PLM write-protect
    assert_eq!(payload.total_bytes, 16);
}

#[test]
fn test_engine_plan_and_dry_run_execution() {
    let rtx3090 = find_recipe(0x2204).unwrap();
    let dev = PciDevice {
        dbdf: "0000:01:00.0".to_string(),
        vendor_id: 0x10DE,
        device_id: 0x2204,
        tier: neutron_gpu::pci::DeviceTier::DiscreteGpu,
        vendor_name: "NVIDIA",
        model_name: "GeForce RTX 3090".to_string(),
        driver: None,
        power_state: "D0".to_string(),
        memory_type: "Dedicated GDDR6X",
        matched_recipe: Some(rtx3090.clone()),
    };

    let plan = UnlockEngine::plan(&dev).expect("Plan should succeed for unbound D0 card");
    assert!(plan.ready_for_execution);
    assert!(plan.warnings.is_empty());
    assert_eq!(plan.mmio_writes.len(), 3);

    // Verify MMIO write values
    assert_eq!(plan.mmio_writes[0].offset, rtx3090.plm_feat_offset);
    assert_eq!(plan.mmio_writes[0].value, 0x0000_0000);

    assert_eq!(plan.mmio_writes[1].offset, rtx3090.ss0_offset);
    assert_eq!(plan.mmio_writes[1].value, UNLOCK_RATE_FULL_MMA);

    assert_eq!(plan.mmio_writes[2].offset, rtx3090.ss1_offset);
    assert_eq!(plan.mmio_writes[2].value, UNLOCK_RATE_FULL_IMLA4);

    // Execute dry run
    let result = UnlockEngine::execute(&plan, rtx3090, true);
    assert!(result.success);
    assert!(result.dry_run);
    assert_eq!(result.executed_changes.len(), 3);
    assert!(result.flr_reset_requested);
}

#[test]
fn test_engine_plan_blocks_bound_driver() {
    let rtx4090 = find_recipe(0x2684).unwrap();
    let dev = PciDevice {
        dbdf: "0000:02:00.0".to_string(),
        vendor_id: 0x10DE,
        device_id: 0x2684,
        tier: neutron_gpu::pci::DeviceTier::DiscreteGpu,
        vendor_name: "NVIDIA",
        model_name: "GeForce RTX 4090".to_string(),
        driver: Some("nvidia".to_string()),
        power_state: "D0".to_string(),
        memory_type: "Dedicated GDDR6X",
        matched_recipe: Some(rtx4090.clone()),
    };

    let plan = UnlockEngine::plan(&dev).expect("Plan generation succeeds with warning");
    assert!(
        !plan.ready_for_execution,
        "Must block execution if driver is bound"
    );
    assert!(!plan.warnings.is_empty());
    assert!(plan.warnings[0].contains("Must unbind"));
}

#[test]
fn test_mdev_profile_discovery() {
    let profiles = MdevManager::list_profiles("0000:01:00.0");
    assert!(!profiles.is_empty());
    assert_eq!(profiles[0].device_api, "vfio-pci");
    assert!(profiles[0].available_instances >= 1);
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

#[test]
fn test_qemu_cmd_and_libvirt_xml() {
    let uuid = "12345678-1234-1234-1234-123456789abc";
    let cmd = MdevManager::generate_qemu_cmd(uuid, "32G", 16, Some("thme-node-1.qcow2"));
    assert!(cmd.contains("qemu-system-x86_64"));
    assert!(cmd.contains("-m 32G"));
    assert!(cmd.contains("-smp 16"));
    assert!(cmd.contains("sysfsdev=/sys/bus/mdev/devices/12345678-1234-1234-1234-123456789abc"));
    assert!(cmd.contains("thme-node-1.qcow2"));

    let xml = MdevManager::generate_libvirt_xml(uuid);
    assert!(xml.contains("<hostdev mode='subsystem' type='mdev'"));
    assert!(xml.contains("<address uuid='12345678-1234-1234-1234-123456789abc'/>"));
}

#[test]
fn test_legacy_pascal_unlock_plan() {
    let gtx1080ti = find_recipe(0x1b06).unwrap();
    let dev = PciDevice {
        dbdf: "0000:03:00.0".to_string(),
        vendor_id: 0x10DE,
        device_id: 0x1b06,
        tier: neutron_gpu::pci::DeviceTier::DiscreteGpu,
        vendor_name: "NVIDIA",
        model_name: "GeForce GTX 1080 Ti".to_string(),
        driver: None,
        power_state: "D0".to_string(),
        memory_type: "Dedicated GDDR5X",
        matched_recipe: Some(gtx1080ti.clone()),
    };

    let plan = UnlockEngine::plan(&dev).unwrap();
    assert!(plan.ready_for_execution);
    assert_eq!(plan.mmio_writes.len(), 2);
    assert_eq!(
        plan.mmio_writes[0].target_name,
        "LEGACY_VGPU_PCI_DEVICE_MAP"
    );
    assert_eq!(
        plan.mmio_writes[1].target_name,
        "NVENC_CONCURRENT_SESSION_LIMIT"
    );
}
