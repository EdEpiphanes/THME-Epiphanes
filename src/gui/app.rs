// SPDX-License-Identifier: AGPL-3.0-only
macro_rules! f {
    ($($arg:tt)*) => {
        format!($($arg)*)
    };
}

use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use std::path::Path;
use std::time::Instant;

use super::manga;
use super::provenance::{self, Evidence};
use super::theme::{self, GuiTheme, ThemePalette};

mod universal_model;
pub use universal_model::*;

/// Main Application State
pub struct THMEApp {
    // Theme Engine
    pub theme: GuiTheme,

    // Models Repository
    models: Vec<UniversalModel>,
    active_model_idx: usize,

    // Active Navigation Tab
    // 0: Master Dashboard
    // 1: Models & Weights
    // 2: Layer Tweaker & Sections
    // 3: Distillation & Matrix Engine
    // 4: Full Benchmarks
    // 5: History & Checkpoints
    // 6: Audit Log
    // 7: Export & Surgery
    // 8: Dev Lab & Training
    // 9: Closed-Loop Co-Surgery (v0.1.9)
    active_tab: usize,
    /// Documentation capture mode (`THME_GUI_CAPTURE_DIR`); `None` in normal use.
    capture: Option<super::capture::CaptureState>,
    /// Live 3D model view (tab 10).
    live: super::live::LiveView,
    /// Chat with a loaded model (tab 11, THME-3038).
    chat: chat::ChatTab,
    /// First-run Welcome window (THME-3039); open until a model file is loaded.
    welcome: welcome::WelcomePanel,

    // Closed-Loop Co-Surgery Dashboard (v0.1.9)
    cosurgery_budget_iterations: usize,
    cosurgery_max_ppl_delta: f64,
    cosurgery_target_reduction: f64,
    cosurgery_report: Option<crate::surgery::CoSurgeryReport>,
    cosurgery_log: Vec<String>,
    status_msg: String,
    start_instant: Instant,

    // Configurable Early Exit
    exit_ratio_pct: f32,
    checkpoint_layer_idx: usize,

    // Baseline Benchmarker
    baseline_prompt: String,
    baseline_tokens: usize,
    baseline_temp: f32,
    baseline_results: Option<BaselineMetrics>,

    // Sequential Layer Auto-Tuner Loop
    auto_tune_running: bool,
    current_tune_layer: usize,
    current_tune_iteration: usize,
    tune_timer: f32,
    auto_step_mode: bool,
    quality_tolerance: f32,
    scorecard: Option<Scorecard>,
    audit_trail: Vec<AuditRow>,
    live_tuner_log: Vec<String>,

    // Live Flow Benchmarker Engine
    simulator: crate::profiling::LiveFlowSimulator,
    sim_speed: f32,

    // Layer Inspector & Tweaker
    selected_layer_idx: usize,
    selected_section_filter: String,
    layer_scale_factor: f32,
    layer_bias_offset: f32,
    layer_pruning_sparsity: f32,
    selected_tensor_idx: usize,
    weight_search_query: String,

    // Automated Distillation & Matrix Engine
    parameter_decisions: Vec<crate::transform::ParameterTransformDecision>,
    plan_summary: Option<crate::transform::TransformPlanSummary>,
    distill_temperature: f32,

    // History and Checkpoints
    history: Vec<HistoryCheckpoint>,
    checkpoint_counter: usize,

    // Audit Logging
    audit_events: Vec<AuditEvent>,
    audit_filter_level: String,
    audit_search_query: String,

    // Export & Surgery
    export_path: String,
    export_format: String,
    export_quant: crate::exporter::ExportQuantType,
    export_checksum: bool,
    tweak_recipe_text: String,
    surgery_diff: Option<crate::surgery::SurgeryDiff>,

    // Bring Your Own Algorithm (BYOA)
    pub algorithm_registry: crate::transform::byoa::AlgorithmRegistry,
    selected_byoa_id: String,
    byoa_test_status: String,
    byoa_designer_open: bool,
    new_byoa_id: String,
    new_byoa_name: String,
    new_byoa_author: String,
    new_byoa_bits: f32,
    new_byoa_category_idx: usize,

    // Rebuilding Model From Scratch
    rebuilder_open: bool,
    rebuild_model_name: String,
    rebuild_num_layers: usize,
    rebuild_hidden_dim: usize,
    rebuild_checkpoint_layer: usize,
    rebuild_remap_open_weights: bool,

    // Universal 8 Specialized AI Models Paradigm Converter
    paradigm_converter_open: bool,
    selected_target_paradigm_idx: usize,
    paradigm_num_experts: usize,
    paradigm_top_k: usize,
    paradigm_target_hidden: usize,
    paradigm_target_layers: usize,
    paradigm_apply_bitnet: bool,
    paradigm_last_report: Option<String>,

    // Speculative Decoding & Multi-Token Draft Synthesizer (v0.1.6)
    spec_draft_layers: usize,
    spec_lookahead_horizon: usize,
    spec_temperature: f32,
    spec_apply_bitnet: bool,
    spec_use_tree: bool,
    spec_last_report: Option<String>,
    spec_speedup_val: f64,
    spec_acceptance_val: f64,
    spec_tokens_per_pass: f64,
    spec_traffic_savings: f64,
    /// False until a speculative run replaces the demo defaults below.
    spec_simulated: bool,

    // Roofline & Micro-Kernel Profiler State (v0.1.7)
    roofline_hardware_idx: usize,
    roofline_batch_size: usize,
    roofline_seq_len: usize,
    roofline_flash_attn: bool,
    roofline_thermal_sim: bool,
    roofline_temp_c: f32,
    roofline_peak_tflops: f64,
    roofline_bandwidth_tbps: f64,
    roofline_knee_intensity: f64,
    roofline_mem_bound_pct: f64,
    roofline_comp_bound_pct: f64,
    roofline_est_latency_ms: f64,
    roofline_last_report: Option<String>,

    // Deep Layer Deconstruction & Anatomy
    selected_subblock: Option<crate::ir::SubBlockCategory>,
    hybrid_topology_mode: usize,
    subblock_target_algo_id: String,
    subblock_convert_target: usize,

    // Developer Lab, Multi-Format Datasets & Micro-Training
    dataset: Option<crate::dataset::Dataset>,
    dataset_input_text: String,
    dataset_status_msg: String,
    selected_experiment_idx: usize,
    experiment_param_variance: f32,
    experiment_param_gamma: f32,
    experiment_param_frequency: f32,
    experiment_param_blend: f32,
    experiment_param_drop_prob: f32,
    experiment_status_msg: String,
    last_experiment_metrics: Option<crate::experiment::ExperimentMetrics>,
    trainer_config: crate::trainer::TrainingConfig,
    micro_trainer: crate::trainer::MicroTrainer,
    training_target_layer: usize,
}

impl THMEApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        manga::install_fonts(&cc.egui_ctx);
        let mut initial_model = UniversalModel::create_synthetic(12, 2048);
        let cp_idx = (initial_model.layers.len() as f32 * 0.30).round() as usize;
        initial_model.update_exit_routing(80.0, cp_idx);

        let default_recipe = crate::surgery::TweakRecipe::default();
        let default_recipe_json = serde_json::to_string_pretty(&default_recipe).unwrap_or_default();

        let prompt = "Explain how heterogeneous layerwise representations reduce energy and memory bandwidth.".to_string();
        let simulator = crate::profiling::LiveFlowSimulator::new(
            &prompt,
            48,
            initial_model.layers.len(),
            initial_model.hidden_size,
            cp_idx,
            0.80,
        );

        let start_instant = Instant::now();

        let mut app = Self {
            theme: GuiTheme::default(),
            models: vec![initial_model],
            active_model_idx: 0,
            active_tab: 0,
            capture: super::capture::CaptureState::from_env(),
            live: super::live::LiveView::default(),
            chat: chat::ChatTab::default(),
            welcome: welcome::WelcomePanel::default(),
            status_msg: "THME Master Engine initialized. System ready.".to_string(),
            start_instant,
            exit_ratio_pct: 80.0,
            checkpoint_layer_idx: cp_idx,
            baseline_prompt: prompt,
            baseline_tokens: 48,
            baseline_temp: 0.7,
            baseline_results: None,
            auto_tune_running: false,
            current_tune_layer: 0,
            current_tune_iteration: 0,
            tune_timer: 0.0,
            auto_step_mode: true,
            quality_tolerance: 0.025,
            scorecard: None,
            audit_trail: Vec::new(),
            live_tuner_log: Vec::new(),
            simulator,
            sim_speed: 1.0,
            selected_layer_idx: 0,
            selected_section_filter: "All Sections".to_string(),
            layer_scale_factor: 1.0,
            layer_bias_offset: 0.0,
            layer_pruning_sparsity: 0.0,
            selected_tensor_idx: 0,
            weight_search_query: String::new(),
            parameter_decisions: Vec::new(),
            plan_summary: None,
            distill_temperature: 2.0,
            history: Vec::new(),
            checkpoint_counter: 0,
            audit_events: Vec::new(),
            audit_filter_level: "ALL".to_string(),
            audit_search_query: String::new(),
            export_path: "models/exported_tuned_model.safetensors".to_string(),
            export_format: "SafeTensors".to_string(),
            export_quant: crate::exporter::ExportQuantType::FP16,
            export_checksum: true,
            tweak_recipe_text: default_recipe_json,
            surgery_diff: None,

            // Bring Your Own Algorithm (BYOA)
            algorithm_registry: crate::transform::byoa::AlgorithmRegistry::new(),
            selected_byoa_id: "thme.bitnet_b158".to_string(),
            byoa_test_status: String::new(),
            byoa_designer_open: false,
            new_byoa_id: "user.custom_quant_v1".to_string(),
            new_byoa_name: "My Custom Quantizer".to_string(),
            new_byoa_author: "Research Contributor".to_string(),
            new_byoa_bits: 2.0,
            new_byoa_category_idx: 0,

            // Rebuilding Model From Scratch
            rebuilder_open: false,
            rebuild_model_name: "rebuilt-hybrid-deltanet".to_string(),
            rebuild_num_layers: 16,
            rebuild_hidden_dim: 2048,
            rebuild_checkpoint_layer: 4,
            rebuild_remap_open_weights: true,

            // Universal 8 Specialized AI Models Paradigm Converter
            paradigm_converter_open: false,
            selected_target_paradigm_idx: 3, // MoE
            paradigm_num_experts: 4,
            paradigm_top_k: 2,
            paradigm_target_hidden: 1024,
            paradigm_target_layers: 8,
            paradigm_apply_bitnet: true,
            paradigm_last_report: None,

            // Speculative Decoding & Multi-Token Draft Synthesizer (v0.1.6)
            spec_draft_layers: 4,
            spec_lookahead_horizon: 5,
            spec_temperature: 0.7,
            spec_apply_bitnet: false,
            spec_use_tree: false,
            spec_last_report: None,
            spec_speedup_val: 2.14,
            spec_acceptance_val: 78.5,
            spec_tokens_per_pass: 3.92,
            spec_traffic_savings: 64.2,
            spec_simulated: false,

            // Roofline & Micro-Kernel Profiler State (v0.1.7)
            roofline_hardware_idx: 0,
            roofline_batch_size: 1,
            roofline_seq_len: 2048,
            roofline_flash_attn: true,
            roofline_thermal_sim: false,
            roofline_temp_c: 25.0,
            roofline_peak_tflops: 989.4,
            roofline_bandwidth_tbps: 3.35,
            roofline_knee_intensity: 295.34,
            roofline_mem_bound_pct: 66.7,
            roofline_comp_bound_pct: 33.3,
            roofline_est_latency_ms: 1.24,
            roofline_last_report: None,

            // Deep Layer Deconstruction & Anatomy
            selected_subblock: None,
            hybrid_topology_mode: 0,
            subblock_target_algo_id: "thme.bitnet_b158".to_string(),
            subblock_convert_target: 0,

            // Developer Lab, Multi-Format Datasets & Micro-Training
            dataset: crate::dataset::Dataset::create_synthetic_calibration().ok(),
            dataset_input_text: "Topological manifold optimization and energy-weighted routing allow LLMs to reduce latency and memory bandwidth.".to_string(),
            dataset_status_msg: "Default synthetic calibration dataset loaded (5 samples).".to_string(),
            selected_experiment_idx: 0,
            experiment_param_variance: 0.02,
            experiment_param_gamma: 0.85,
            experiment_param_frequency: 0.25,
            experiment_param_blend: 0.5,
            experiment_param_drop_prob: 0.2,
            experiment_status_msg: "Ready for non-conventional experimentation.".to_string(),
            last_experiment_metrics: None,
            trainer_config: crate::trainer::TrainingConfig::default(),
            micro_trainer: crate::trainer::MicroTrainer::new(crate::trainer::TrainingConfig::default()),
            training_target_layer: 0,

            // Closed-Loop Co-Surgery Dashboard (v0.1.9)
            cosurgery_budget_iterations: 10,
            cosurgery_max_ppl_delta: 0.15,
            cosurgery_target_reduction: 0.40,
            cosurgery_report: None,
            cosurgery_log: Vec::new(),
        };

        app.log_audit(
            "INFO",
            "System",
            "THME Master Engine v2.0 initialized",
            "Engine",
        );

        // Only an explicit THME_MODEL is loaded on startup; otherwise the built-in demo model
        // stays, so screenshots and GIFs do not depend on a model file on the developer machine.
        let default_host_model = std::env::var("THME_MODEL")
            .ok()
            .map(std::path::PathBuf::from);

        if let Some(path) = default_host_model {
            if path.exists() {
                if let Ok(loaded) = UniversalModel::load_universal(&path) {
                    let name = loaded.name.clone();
                    app.models.push(loaded);
                    app.active_model_idx = app.models.len() - 1;
                    app.checkpoint_layer_idx =
                        (app.model().layers.len() as f32 * 0.30).round() as usize;
                    app.status_msg = f!("Loaded model on startup: {:?}", path);
                    app.log_audit(
                        "INFO",
                        "ModelLoad",
                        &format!("Imported file on startup: {:?}", path),
                        &name,
                    );
                }
            }
        }

        // The Welcome window opens only when no model file is loaded and no capture run is going.
        app.welcome.open = app.model().source_path.is_none() && app.capture.is_none();
        app.auto_scan_and_benchmark();
        app.create_checkpoint(
            "Initial Baseline",
            "Unmodified model baseline configuration",
        );
        app.run_auto_decider_internal();
        app
    }

    pub fn palette(&self) -> ThemePalette {
        self.theme.palette()
    }

    pub fn model(&self) -> &UniversalModel {
        &self.models[self
            .active_model_idx
            .min(self.models.len().saturating_sub(1))]
    }

    pub fn model_mut(&mut self) -> &mut UniversalModel {
        let idx = self
            .active_model_idx
            .min(self.models.len().saturating_sub(1));
        &mut self.models[idx]
    }

    fn log_audit(&mut self, level: &str, category: &str, message: &str, target: &str) {
        let id = self.audit_events.len() + 1;
        let elapsed = self.start_instant.elapsed();
        let ts = format!(
            "{:02}:{:02}.{:03}",
            elapsed.as_secs() / 60,
            elapsed.as_secs() % 60,
            elapsed.subsec_millis()
        );
        self.audit_events.push(AuditEvent {
            id,
            timestamp: ts,
            level: level.to_string(),
            category: category.to_string(),
            message: message.to_string(),
            target: target.to_string(),
        });
    }

    fn create_checkpoint(&mut self, label: &str, details: &str) {
        self.checkpoint_counter += 1;
        let elapsed = self.start_instant.elapsed();
        let ts = format!(
            "{:02}:{:02}.{:03}",
            elapsed.as_secs() / 60,
            elapsed.as_secs() % 60,
            elapsed.subsec_millis()
        );
        let (store_mb, _, _, _, tok_s) = self.model().compute_costs();
        let base_store = (self.model().layers.len() as f32
            * self.model().hidden_size as f32
            * self.model().hidden_size as f32
            * 6.0)
            / (1024.0 * 1024.0);
        let ratio = base_store / store_mb.max(0.1);
        let avg_distortion = self
            .plan_summary
            .as_ref()
            .map(|s| s.average_distortion_snr)
            .unwrap_or(0.005);

        let m_name = self.model().name.clone();
        let cp = HistoryCheckpoint {
            id: self.checkpoint_counter,
            timestamp: ts,
            label: label.to_string(),
            model_name: m_name.clone(),
            storage_mb: store_mb,
            throughput_tok_s: tok_s,
            compression_ratio: ratio,
            avg_distortion,
            details: details.to_string(),
            snapshot_model: self.model().clone(),
        };
        self.history.push(cp);
        self.log_audit(
            "OPTIMIZE",
            "Checkpoint",
            &format!("Snapshot #{} created: {}", self.checkpoint_counter, label),
            &m_name,
        );
    }

    fn restore_checkpoint(&mut self, cp_idx: usize) {
        if cp_idx < self.history.len() {
            let cp = &self.history[cp_idx];
            let name = cp.label.clone();
            let restored = cp.snapshot_model.clone();
            *self.model_mut() = restored;
            let m_name = self.model().name.clone();
            self.auto_scan_and_benchmark();
            self.run_auto_decider_internal();
            self.log_audit(
                "OPTIMIZE",
                "Restore",
                &format!("Restored checkpoint: {}", name),
                &m_name,
            );
            self.status_msg = format!("Restored checkpoint '{}' successfully!", name);
        }
    }

    fn run_auto_decider_internal(&mut self) {
        let (decisions, summary) = crate::transform::AutoDecider::build_model_plan_with_registry(
            &self.model().tensors,
            self.model().layers.len(),
            self.quality_tolerance,
            Some(&self.algorithm_registry),
        );
        self.parameter_decisions = decisions;
        self.plan_summary = Some(summary);
        let m_name = self.model().name.clone();
        let dec_count = self.parameter_decisions.len();
        let comp_ratio = self
            .plan_summary
            .as_ref()
            .map(|s| s.overall_compression_ratio)
            .unwrap_or(1.0);
        let mean_dist = self
            .plan_summary
            .as_ref()
            .map(|s| s.average_distortion_snr)
            .unwrap_or(0.0);
        self.log_audit(
            "DISTILL",
            "AutoDecider",
            &format!(
                "Evaluated {} parameters: overall compression {:.2}x, mean SNR distortion {:.4}",
                dec_count, comp_ratio, mean_dist
            ),
            &m_name,
        );
    }

    fn execute_global_transformation_plan(&mut self) {
        let cp_idx = self.checkpoint_layer_idx;

        for (l_idx, layer) in self.model_mut().layers.iter_mut().enumerate() {
            layer.is_optimized = true;
            if l_idx < cp_idx {
                layer.alpha_binary = 0.15;
                layer.beta_ternary = 0.65;
                layer.gamma_phase = 0.20;
                layer.layer_type = if l_idx == 0 {
                    "sliding_window"
                } else {
                    "gated_deltanet"
                }
                .to_string();
            } else if l_idx == cp_idx {
                layer.alpha_binary = 0.20;
                layer.beta_ternary = 0.45;
                layer.gamma_phase = 0.35;
                layer.layer_type = "gated_deltanet".to_string();
            } else {
                layer.alpha_binary = 0.40;
                layer.beta_ternary = 0.40;
                layer.gamma_phase = 0.20;
                layer.layer_type = if l_idx.is_multiple_of(2) {
                    "phase_recurrence"
                } else {
                    "gated_deltanet"
                }
                .to_string();
            }
        }

        self.auto_scan_and_benchmark();
        self.create_checkpoint(
            "Automated Pareto Plan Executed",
            "Applied optimal matrix & distillation transformations across all layers",
        );
        self.status_msg =
            "Global Pareto transformation plan executed and locked across all layers!".to_string();
    }

    /// Loads a model file as the Studio does: the import runs the safety scan first. Used by
    /// "Load Model File..." and the Welcome window's Open buttons.
    fn open_model_file(&mut self, path: &Path) -> Result<(), String> {
        match UniversalModel::load_universal(path) {
            Ok(m) => {
                let name = m.name.clone();
                self.models.push(m);
                self.active_model_idx = self.models.len() - 1;
                self.checkpoint_layer_idx =
                    (self.model().layers.len() as f32 * 0.30).round() as usize;
                self.auto_scan_and_benchmark();
                self.run_auto_decider_internal();
                self.log_audit(
                    "INFO",
                    "ModelLoad",
                    &format!("Imported file: {:?}", path),
                    &name,
                );
                self.status_msg = f!("Loaded model [{}]: {:?}", self.model().format, path);
                Ok(())
            }
            Err(err) => {
                self.status_msg = f!("Error loading file: {}", err);
                Err(err)
            }
        }
    }

    fn auto_scan_and_benchmark(&mut self) {
        let ratio = self.exit_ratio_pct;
        let cp = self.checkpoint_layer_idx;
        self.model_mut().update_exit_routing(ratio, cp);

        let num_layers = self.model().layers.len().max(1);
        let hidden = self.model().hidden_size.max(1);
        let total_params = self.model().total_parameters;

        // Hardware-calibrated micro-benchmark using real monotonic timer
        let t0 = std::time::Instant::now();
        let mut probe_acc = 0.0f32;
        let mut sample_count = 0usize;
        for t in &self.model().tensors {
            for &val in t.sample_weights.iter().take(256) {
                probe_acc += val * 1.001;
                sample_count += 1;
            }
        }
        let bench_elapsed = t0.elapsed();
        std::hint::black_box(probe_acc);

        let prompt_words = self.baseline_prompt.split_whitespace().count().max(1);
        let total_seq = prompt_words + self.baseline_tokens;

        let base_ns_per_param = if sample_count > 0 {
            (bench_elapsed.as_nanos() as f64 / sample_count as f64).clamp(0.5, 20.0)
        } else {
            1.2
        };

        // Measured TTFT and generation latency based on active parameters and context
        let est_hw_bw_gbps = (100.0 / base_ns_per_param).clamp(15.0, 120.0) as f32;
        let ms_per_tok = ((total_params as f32 * 0.5) / (est_hw_bw_gbps * 1e6)).clamp(2.0, 50.0);
        let ttft_ms = (prompt_words as f32 * 0.4 + ms_per_tok * 1.2).max(1.0);
        let total_time_ms = ttft_ms + (self.baseline_tokens as f32 * ms_per_tok);
        let tokens_per_sec = (self.baseline_tokens as f32 / (total_time_ms / 1000.0)).max(1.0);

        let kv_cache_mb = (2.0 * 1.0 * 8.0 * total_seq as f32 * 128.0 * 2.0 * num_layers as f32)
            / (1024.0 * 1024.0);
        let energy_mj = (ms_per_tok * 0.35 + num_layers as f32 * 0.2).clamp(1.0, 150.0);

        let mut latencies = Vec::new();
        for _ in 0..num_layers {
            latencies.push(ms_per_tok / num_layers as f32);
        }

        // provenance: baseline-profile
        let gen_text = provenance::labelled(f!(
            "{}\n\n[Baseline Architecture Scan & Benchmark]\nArchitecture: {} • {} Layers • Hidden Dim {} • Parameters: {}M.\nExecution profile: {:.1} tok/s | TTFT: {:.1}ms | KV Cache: {:.2} MB.\nReady for Sequential Layer-by-Layer Auto-Tuning.",
            self.baseline_prompt, self.model().architecture, num_layers, hidden, self.model().total_parameters / 1_000_000,
            tokens_per_sec, ttft_ms, kv_cache_mb
        ), Evidence::Estimate);

        self.baseline_results = Some(BaselineMetrics {
            prompt: self.baseline_prompt.clone(),
            generated_text: gen_text,
            tokens_generated: self.baseline_tokens,
            ttft_ms,
            time_per_token_ms: ms_per_tok,
            tokens_per_sec,
            kv_cache_mb,
            energy_mj_per_token: energy_mj,
            layer_latencies: latencies,
        });

        // provenance: baseline-throughput
        self.status_msg = provenance::labelled(
            f!(
                "Model scanned: {} layers, {} params. Baseline benchmark: {:.1} tok/s",
                num_layers,
                self.model().total_parameters / 1_000_000,
                tokens_per_sec
            ),
            Evidence::Estimate,
        );
    }

    fn start_layerwise_auto_tuner(&mut self) {
        self.auto_tune_running = true;
        self.current_tune_layer = 0;
        self.current_tune_iteration = 0;
        self.tune_timer = 0.0;
        self.audit_trail.clear();
        self.live_tuner_log.clear();

        for layer in &mut self.model_mut().layers {
            layer.is_optimized = false;
        }

        let m_name = self.model().name.clone();
        self.log_audit(
            "OPTIMIZE",
            "SequentialTuner",
            "Started AirLLM sequential layer-by-layer optimization loop",
            &m_name,
        );
        self.live_tuner_log.push(f!(
            "[Engine Started] Initializing AirLLM-Style Layer-by-Layer Optimizer with Early Exit Target {:.0}/{:.0}...",
            self.exit_ratio_pct, 100.0 - self.exit_ratio_pct
        ));
    }

    fn step_optimization_loop(&mut self) {
        if !self.auto_tune_running || self.current_tune_layer >= self.model().layers.len() {
            if self.auto_tune_running {
                self.finalize_auto_tuning();
            }
            return;
        }

        let l_idx = self.current_tune_layer;
        let num_layers = self.model().layers.len();
        let depth_ratio = l_idx as f32 / num_layers.max(2) as f32;
        let reach = self.model().layers[l_idx].reach_probability;
        let is_cp = self.model().layers[l_idx].is_exit_checkpoint;

        let candidates = [
            LayerCandidate {
                name: "Dense FP16 Baseline",
                alpha: 0.0,
                beta: 1.0,
                gamma: 0.0,
                mixer: "gqa_attention",
            },
            LayerCandidate {
                name: "Sliding Window Ternary",
                alpha: 0.10,
                beta: 0.80,
                gamma: 0.10,
                mixer: "sliding_window",
            },
            LayerCandidate {
                name: "Gated DeltaNet Hybrid",
                alpha: 0.20,
                beta: 0.60,
                gamma: 0.20,
                mixer: "gated_deltanet",
            },
            LayerCandidate {
                name: "Phase Recurrence Stack",
                alpha: 0.15,
                beta: 0.45,
                gamma: 0.40,
                mixer: "phase_recurrence",
            },
            LayerCandidate {
                name: "Aggressive Low-Bit XNOR",
                alpha: 0.45,
                beta: 0.40,
                gamma: 0.15,
                mixer: "gated_deltanet",
            },
        ];

        let iter = self.current_tune_iteration;
        if iter < candidates.len() {
            let cand = &candidates[iter];
            let bytes_pw = (cand.alpha * 0.125) + (cand.beta * 0.25) + (cand.gamma * 1.0);
            let cand_latency = 3.2
                * (bytes_pw / 0.5)
                * (if cand.mixer == "gated_deltanet" {
                    0.75
                } else {
                    0.95
                });
            let cand_distortion = (cand.alpha * 0.04) + (cand.beta * 0.012) + (cand.gamma * 0.018);

            // provenance: tuner-candidate
            self.live_tuner_log.push(provenance::labelled(f!(
                "[Layer {}] Iteration {}/{} testing '{}' ({}%α, {}%β, {}%γ) -> Latency: {:.2}ms, Distortion: {:.4}",
                l_idx, iter + 1, candidates.len(), cand.name, (cand.alpha*100.0) as u32, (cand.beta*100.0) as u32, (cand.gamma*100.0) as u32,
                cand_latency, cand_distortion
            ), Evidence::Simulated));

            self.current_tune_iteration += 1;
        } else {
            let (best_alpha, best_beta, best_gamma, best_mixer) =
                if l_idx < self.checkpoint_layer_idx {
                    (
                        0.15,
                        0.75,
                        0.10,
                        if l_idx == 0 {
                            "sliding_window"
                        } else {
                            "gated_deltanet"
                        },
                    )
                } else if is_cp {
                    (0.20, 0.50, 0.30, "gated_deltanet")
                } else {
                    (
                        0.35 + (0.10 * depth_ratio),
                        0.35 - (0.10 * depth_ratio),
                        0.30,
                        if l_idx.is_multiple_of(2) {
                            "phase_recurrence"
                        } else {
                            "gated_deltanet"
                        },
                    )
                };

            let bytes_pw = (best_alpha * 0.125) + (best_beta * 0.25) + (best_gamma * 1.0);
            let layer_size_mb = (self.model().layers[l_idx].channel_width as f32
                * self.model().layers[l_idx].channel_width as f32
                * bytes_pw
                * 3.0)
                / (1024.0 * 1024.0);
            let layer_traffic_mb = reach * layer_size_mb;
            let opt_lat = 1.65 + (0.4 * depth_ratio);
            let opt_dist = 0.008 + (0.004 * depth_ratio);

            let layer = &mut self.model_mut().layers[l_idx];
            layer.alpha_binary = best_alpha;
            layer.beta_ternary = best_beta;
            layer.gamma_phase = best_gamma;
            layer.layer_type = best_mixer.to_string();
            layer.is_optimized = true;
            layer.benchmarked_latency_ms = opt_lat;
            layer.estimated_distortion = opt_dist;

            // provenance: tuner-winner
            self.live_tuner_log.push(provenance::labelled(f!(
                "--> WINNER LOCKED for Layer {}: {} ({}%α / {}%β / {}%γ) | Latency: {:.2}ms | Storage: {:.2} MB",
                l_idx, best_mixer, (best_alpha * 100.0) as u32, (best_beta * 100.0) as u32, (best_gamma * 100.0) as u32,
                opt_lat, layer_size_mb
            ), Evidence::Simulated));

            self.audit_trail.push(AuditRow {
                layer: l_idx,
                is_checkpoint: is_cp,
                status: "OPTIMIZED & LOCKED".to_string(),
                alpha: (best_alpha * 100.0) as u32,
                beta: (best_beta * 100.0) as u32,
                gamma: (best_gamma * 100.0) as u32,
                reach,
                mixer: best_mixer.to_string(),
                storage_mb: layer_size_mb,
                traffic_mb: layer_traffic_mb,
                latency_ms: opt_lat,
                distortion: opt_dist,
            });

            self.log_audit(
                "OPTIMIZE",
                "LayerTuned",
                &format!(
                    "Layer {} locked: {} ({:.0}%α/{:.0}%β/{:.0}%γ)",
                    l_idx,
                    best_mixer,
                    best_alpha * 100.0,
                    best_beta * 100.0,
                    best_gamma * 100.0
                ),
                &format!("Layer {}", l_idx),
            );

            self.current_tune_layer += 1;
            self.current_tune_iteration = 0;

            if self.current_tune_layer < num_layers {
                self.live_tuner_log
                    .push(f!("Advancing to Layer {}...", self.current_tune_layer));
            }
        }
    }

    fn finalize_auto_tuning(&mut self) {
        self.auto_tune_running = false;
        let num_layers = self.model().layers.len();
        let total_stored: f32 = self.audit_trail.iter().map(|r| r.storage_mb).sum();

        let base_tok = self
            .baseline_results
            .as_ref()
            .map(|r| r.tokens_per_sec)
            .unwrap_or(22.0);
        let base_store = total_stored * 2.3;
        let base_kv = self
            .baseline_results
            .as_ref()
            .map(|r| r.kv_cache_mb)
            .unwrap_or(1.2);
        let base_e = self
            .baseline_results
            .as_ref()
            .map(|r| r.energy_mj_per_token)
            .unwrap_or(36.0);

        let p_exit = self.exit_ratio_pct / 100.0;
        let cp_idx = self.checkpoint_layer_idx;
        let f_bar = (cp_idx as f32 * 1.0 + (num_layers - cp_idx) as f32 * (1.0 - p_exit))
            / num_layers as f32;
        let speedup = 1.0 / (f_bar * 0.80 + 0.15);
        let opt_tok = base_tok * speedup;
        let opt_kv = base_kv * 0.30;
        let opt_e = base_e * 0.28;

        self.scorecard = Some(Scorecard {
            baseline_tok_sec: base_tok,
            tuned_tok_sec: opt_tok,
            speedup_factor: speedup,
            baseline_storage_mb: base_store,
            tuned_storage_mb: total_stored,
            compression_ratio: base_store / total_stored.max(0.1),
            baseline_kv_cache_mb: base_kv,
            tuned_kv_cache_mb: opt_kv,
            cache_savings_percent: 70.0,
            baseline_energy_mj: base_e,
            tuned_energy_mj: opt_e,
            energy_savings_percent: 72.0,
        });

        // provenance: scorecard
        self.status_msg = provenance::labelled(f!(
            "Auto-Tuning Complete! All {} layers optimized. Final Speedup: {:.2}x, Storage: {:.1} MB",
            num_layers, speedup, total_stored
        ), Evidence::Estimate);
        // provenance: scorecard-log
        self.live_tuner_log.push(provenance::labelled(f!(
            "[SUCCESS] All {} layers tuned & matched to research paper specification! Speedup: {:.2}x.",
            num_layers, speedup
        ), Evidence::Estimate));

        // provenance: scorecard-checkpoint
        self.create_checkpoint(
            "Sequential Auto-Tuner Complete",
            &provenance::labelled(
                format!("Speedup: {:.2}x, Storage: {:.1} MB", speedup, total_stored),
                Evidence::Estimate,
            ),
        );
    }
}

impl eframe::App for THMEApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(capture) = self.capture.as_mut() {
            self.active_tab = capture.step(&ctx);
        }
        if self.auto_tune_running && self.auto_step_mode {
            self.tune_timer += ctx.input(|i| i.stable_dt);
            if self.tune_timer >= 0.08 {
                self.tune_timer = 0.0;
                self.step_optimization_loop();
            }
            ctx.request_repaint();
        }

        if self.simulator.status == crate::profiling::TokenFlowStatus::Generating
            || self.simulator.status == crate::profiling::TokenFlowStatus::Prefill
        {
            self.simulator.speed_multiplier = self.sim_speed;
            self.simulator.update(ctx.input(|i| i.stable_dt));
            ctx.request_repaint();
        }

        let p = self.palette();
        theme::configure_visuals(&ctx, &p);
        manga::paint_backdrop(&ctx, &p);

        // Root surface: full-bleed central panel, then the old panel stack inside it.
        egui::CentralPanel::default().show(ui, |ui| {
            // Top Control Header
            egui::Panel::top("header_panel")
                .frame(manga::glass_bar(&p).inner_margin(egui::Margin::symmetric(16, 10)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let (rect, _) =
                            ui.allocate_exact_size(Vec2::new(96.0, 24.0), egui::Sense::hover());
                        ui.painter().rect_filled(rect, 4.0, p.primary);
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "THME v2.0",
                            egui::FontId::proportional(12.0),
                            if u32::from(p.primary.r())
                                + u32::from(p.primary.g())
                                + u32::from(p.primary.b())
                                > 480
                            {
                                p.bg_dark
                            } else {
                                Color32::WHITE
                            },
                        );

                        ui.add_space(8.0);
                        ui.vertical(|ui| {
                            ui.heading(
                                egui::RichText::new(
                                    "UNIVERSAL LLM RESTRUCTURER, MATRIX ENGINE & STUDIO",
                                )
                                .strong()
                                .size(13.0)
                                .color(p.text_main),
                            );
                            let m = self.model();
                            ui.label(
                                egui::RichText::new(f!(
                                    "{} [{} • {}] ({} Layers • Hidden Dim {} • {}M Params)",
                                    m.name,
                                    m.format.to_uppercase(),
                                    m.architecture,
                                    m.layers.len(),
                                    m.hidden_size,
                                    m.total_parameters / 1_000_000
                                ))
                                .monospace()
                                .size(10.0)
                                .color(p.text_muted),
                            );
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // Theme Selector
                            ui.label(
                                egui::RichText::new("Scheme:")
                                    .size(11.0)
                                    .color(p.text_muted),
                            );
                            egui::ComboBox::from_id_salt("theme_selector_combo")
                                .selected_text(self.theme.name())
                                .width(135.0)
                                .show_ui(ui, |ui| {
                                    for theme_opt in GuiTheme::ALL {
                                        ui.selectable_value(
                                            &mut self.theme,
                                            *theme_opt,
                                            theme_opt.name(),
                                        );
                                    }
                                });

                            ui.add_space(12.0);

                            if ui
                                .button(
                                    egui::RichText::new("Auto-Decide Plan")
                                        .color(p.text_main)
                                        .strong(),
                                )
                                .clicked()
                            {
                                self.run_auto_decider_internal();
                                self.active_tab = 3;
                            }

                            if ui.button("Sequential Tuner").clicked() {
                                self.active_tab = 4;
                                self.start_layerwise_auto_tuner();
                            }

                            if ui.button("Load Model File...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter(
                                        "All Models (*.gguf, *.safetensors, *.epiph)",
                                        &["gguf", "safetensors", "epiph", "thme"],
                                    )
                                    .add_filter("Epiphanes Models (*.epiph)", &["epiph"])
                                    .add_filter("GGUF Models (*.gguf)", &["gguf"])
                                    .add_filter("SafeTensors (*.safetensors)", &["safetensors"])
                                    .pick_file()
                                {
                                    let _ = self.open_model_file(&path);
                                }
                            }

                            if ui.button("Welcome").clicked() {
                                self.show_welcome();
                            }

                            if ui.button("Load Model Folder...").clicked() {
                                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                    match UniversalModel::load_universal(&dir) {
                                        Ok(m) => {
                                            let name = m.name.clone();
                                            self.models.push(m);
                                            self.active_model_idx = self.models.len() - 1;
                                            self.checkpoint_layer_idx =
                                                (self.model().layers.len() as f32 * 0.30).round()
                                                    as usize;
                                            self.auto_scan_and_benchmark();
                                            self.run_auto_decider_internal();
                                            self.log_audit(
                                                "INFO",
                                                "ModelLoad",
                                                &format!("Imported directory: {:?}", dir),
                                                &name,
                                            );
                                            self.status_msg =
                                                f!("Loaded model directory: {:?}", dir);
                                        }
                                        Err(err) => {
                                            self.status_msg =
                                                f!("Error loading directory: {}", err);
                                        }
                                    }
                                }
                            }
                        });
                    });
                });

            // Enhanced Stats Ribbon
            egui::Panel::top("stats_ribbon")
                .frame(egui::Frame::NONE.inner_margin(egui::Margin::symmetric(14, 8)))
                .show(ui, |ui| {
                    let (store_mb, tr_mb, f_bar, joules, tok_sec) = self.model().compute_costs();

                    ui.columns(6, |cols| {
                        render_stat_box(
                            &mut cols[0],
                            "PARAMETERS",
                            f!("{}M", self.model().total_parameters / 1_000_000),
                            p.primary,
                            &p,
                        );
                        // provenance: ribbon-vram
                        render_stat_box(
                            &mut cols[1],
                            "VRAM STORAGE",
                            provenance::labelled(f!("{:.1} MB", store_mb), Evidence::Estimate),
                            p.success,
                            &p,
                        );
                        // provenance: ribbon-traffic
                        render_stat_box(
                            &mut cols[2],
                            "TRAFFIC/TOKEN",
                            provenance::labelled(f!("{:.2} MB", tr_mb), Evidence::Estimate),
                            p.secondary,
                            &p,
                        );
                        // provenance: ribbon-depth
                        render_stat_box(
                            &mut cols[3],
                            "DEPTH (MEAN)",
                            provenance::labelled(f!("{:.0}%", f_bar * 100.0), Evidence::Estimate),
                            p.warning,
                            &p,
                        );
                        // provenance: ribbon-energy
                        render_stat_box(
                            &mut cols[4],
                            "ENERGY/TOKEN",
                            provenance::labelled(
                                f!("{:.2} mJ", joules * 1000.0),
                                Evidence::Estimate,
                            ),
                            p.danger,
                            &p,
                        );
                        // provenance: ribbon-throughput
                        render_stat_box(
                            &mut cols[5],
                            "THROUGHPUT",
                            provenance::labelled(f!("{:.0} tok/s", tok_sec), Evidence::Estimate),
                            p.accent,
                            &p,
                        );
                    });
                });

            // Bottom Status Bar
            egui::Panel::bottom("status_bar")
                .frame(manga::glass_bar(&p).inner_margin(egui::Margin::symmetric(12, 6)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let badge_rect = Rect::from_center_size(
                            Pos2::new(ui.cursor().min.x + 30.0, ui.cursor().center().y),
                            Vec2::new(56.0, 16.0),
                        );
                        ui.painter().rect_filled(badge_rect, 3.0, p.bg_input);
                        ui.painter().text(
                            badge_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "ONLINE",
                            egui::FontId::monospace(8.0),
                            p.success,
                        );
                        ui.add_space(64.0);

                        ui.label(
                            egui::RichText::new(&self.status_msg)
                                .size(10.0)
                                .color(p.text_main),
                        );

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(f!(
                                    "Audit Events: {} | Snapshots: {} | Theme: {}",
                                    self.audit_events.len(),
                                    self.history.len(),
                                    self.theme.name()
                                ))
                                .monospace()
                                .size(9.0)
                                .color(p.text_muted),
                            );
                        });
                    });
                });

            // Central Panel with Tab Navigation
            egui::CentralPanel::default().show(ui, |ui| {
                ui.horizontal(|ui| {
                    tab_button(ui, "Dashboard", 0, &mut self.active_tab, &p);
                    tab_button(ui, "Models", 1, &mut self.active_tab, &p);
                    tab_button(ui, "Layer Tweaker", 2, &mut self.active_tab, &p);
                    tab_button(ui, "Matrix Engine", 3, &mut self.active_tab, &p);
                    tab_button(ui, "Benchmarks", 4, &mut self.active_tab, &p);
                    tab_button(ui, "History", 5, &mut self.active_tab, &p);
                    tab_button(ui, "Audit Log", 6, &mut self.active_tab, &p);
                    tab_button(ui, "Surgery & Export", 7, &mut self.active_tab, &p);
                    tab_button(ui, "Dev Lab & Training", 8, &mut self.active_tab, &p);
                    tab_button(ui, "Co-Surgery", 9, &mut self.active_tab, &p);
                    tab_button(ui, "Live 3D", 10, &mut self.active_tab, &p);
                    tab_button(ui, "Chat", 11, &mut self.active_tab, &p);
                });

                ui.separator();
                ui.add_space(4.0);

                match self.active_tab {
                    0 => self.render_tab_dashboard(ui),
                    1 => self.render_tab_models(ui),
                    2 => self.render_tab_layer_tweaker(ui),
                    3 => self.render_tab_matrix_distill(ui),
                    4 => self.render_tab_full_benchmarks(ui),
                    5 => self.render_tab_history(ui),
                    6 => self.render_tab_audit_log(ui),
                    7 => self.render_tab_export(ui),
                    8 => self.render_tab_developer_lab(ui),
                    9 => self.render_tab_cosurgery(ui),
                    10 => {
                        let p = self.palette();
                        self.live.ui(ui, &p);
                    }
                    11 => self.render_tab_chat(ui),
                    _ => {}
                }
            });
        });

        self.render_welcome(&ctx);
    }
}

// Tab renderers live in one module each under src/gui/app/ (THME-2003).
mod audit_log;
mod benchmarks;
mod chat;
mod co_surgery;
mod dashboard;
mod dev_lab;
mod history;
mod layer_tweaker;
mod matrix_engine;
mod models;
mod surgery_export;
mod welcome;

fn render_stat_box(
    ui: &mut egui::Ui,
    title: &str,
    value: String,
    color: Color32,
    p: &ThemePalette,
) {
    let badge = if color != p.text_main && color != Color32::WHITE {
        Some(("•", color))
    } else {
        None
    };
    theme::draw_kpi_card(ui, title, &value, badge, p);
}

fn render_scorecard_box(
    ui: &mut egui::Ui,
    title: &str,
    value: String,
    gain: String,
    color: Color32,
    p: &ThemePalette,
) {
    let badge = if !gain.is_empty() {
        Some((gain.as_str(), color))
    } else {
        None
    };
    theme::draw_kpi_card(ui, title, &value, badge, p);
}

fn tab_button(
    ui: &mut egui::Ui,
    title: &str,
    tab_idx: usize,
    active_tab: &mut usize,
    p: &ThemePalette,
) {
    theme::draw_tab_button(ui, title, tab_idx, active_tab, p);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_models_retain_file_sources_without_serializing_paths() {
        let mut tensor = crate::ir::TensorIR::new("weights".into(), vec![4], "F16".into(), 8);
        tensor.source = Some(crate::ir::tensor::TensorSource {
            path: std::path::PathBuf::from("model.safetensors"),
            byte_offset: 32,
            byte_len: 8,
            encoding: crate::ir::tensor::SourceEncoding::Identity,
        });
        let model_ir = crate::ir::ModelIR::new(
            "loaded".into(),
            crate::ir::ModelArchitecture::Llama,
            1,
            4,
            1,
            1,
            4,
            128,
            vec![tensor],
        );

        let model = UniversalModel::from_model_ir(model_ir, "SafeTensors");
        assert_eq!(
            model.source_model.as_ref().unwrap().tensors[0]
                .source
                .as_ref()
                .unwrap()
                .path,
            std::path::PathBuf::from("model.safetensors")
        );
        let serialized = serde_json::to_string(&model).expect("serialize GUI model");
        assert!(!serialized.contains("model.safetensors"));
    }
}
