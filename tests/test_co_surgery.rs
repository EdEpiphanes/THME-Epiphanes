// SPDX-License-Identifier: AGPL-3.0-only
use clap::Parser;
use thme::cli::{Cli, Commands};
use thme::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder};
use thme::surgery::{CoSurgeryBudget, CoSurgeryController, CoSurgeryState};

#[test]
fn co_surgery_cli_parses_defaults() {
    let cli = Cli::try_parse_from(["thme", "co-surgery"]).expect("parse co-surgery with no args");
    assert!(matches!(cli.command, Commands::CoSurgery(_)));
}

#[test]
fn co_surgery_cli_parses_all_flags() {
    let cli = Cli::try_parse_from([
        "thme",
        "co-surgery",
        "--budget-iterations",
        "5",
        "--max-ppl-delta",
        "0.20",
        "--target-reduction",
        "0.30",
        "--json",
    ])
    .expect("parse co-surgery flags");
    let Commands::CoSurgery(args) = cli.command else {
        panic!("expected CoSurgery command");
    };
    assert_eq!(args.budget_iterations, 5);
    assert!((args.max_ppl_delta - 0.20).abs() < 1e-9);
    assert!((args.target_reduction - 0.30).abs() < 1e-9);
    assert!(args.json);
    assert!(args.agent_command.is_none());
}

#[test]
fn co_surgery_alias_works() {
    let cli =
        Cli::try_parse_from(["thme", "cosurgery"]).expect("cosurgery alias should be accepted");
    assert!(matches!(cli.command, Commands::CoSurgery(_)));
}

#[test]
fn co_surgery_controller_completes_on_scaffold() {
    // Verifies the full controller loop runs without panicking and produces a coherent report.
    let bp = ArchitectureBlueprint::new_hybrid_llama("test", 2, 128, 1);
    let model = ModelRebuilder::build_scaffold(&bp);
    let budget = CoSurgeryBudget {
        max_iterations: 3,
        max_ppl_delta: 2.0,
        target_reduction_ratio: 0.50,
        max_consecutive_rollbacks: 5,
    };
    let mut ctrl = CoSurgeryController::new(model, budget);
    let report = ctrl.run_to_completion();

    assert_eq!(report.final_state, CoSurgeryState::Terminated);
    assert!(report.total_iterations >= 1);
    assert!(report.successful_commits + report.total_rollbacks >= 1);
    // PPL delta must stay finite.
    assert!(report.final_ppl.is_finite());
}
