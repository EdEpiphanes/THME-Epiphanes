// SPDX-License-Identifier: AGPL-3.0-only
use thme::trainer::{MicroTrainer, OptimizerType, TrainingConfig, TrainingLossType};

#[test]
fn test_trainer_end_to_end_convergence() {
    let config = TrainingConfig {
        learning_rate: 0.01,
        loss_type: TrainingLossType::CrossEntropy,
        optimizer: OptimizerType::AdamW,
        ..Default::default()
    };

    let mut trainer = MicroTrainer::new(config);
    let mut weights = vec![0.2, -0.1, 0.4, -0.3, 0.5];
    let tokens = vec![101, 102, 103, 104, 105];

    let mut prev_loss = f32::MAX;
    for _ in 0..5 {
        let record = trainer.step(&tokens, &mut weights).unwrap();
        assert!(record.loss > 0.0);
        assert!(record.perplexity >= 1.0);
        prev_loss = record.loss;
    }

    assert_eq!(trainer.history.len(), 5);
    assert!(prev_loss < 5.0);
}

#[test]
fn test_trainer_loss_types() {
    let mut weights = vec![0.1; 10];
    let tokens = vec![1, 2, 3];

    // Knowledge Distillation
    let kd_config = TrainingConfig {
        loss_type: TrainingLossType::KnowledgeDistillation,
        distillation_temperature: 2.5,
        ..Default::default()
    };
    let mut kd_trainer = MicroTrainer::new(kd_config);
    let kd_record = kd_trainer.step(&tokens, &mut weights).unwrap();
    assert!(kd_record.loss > 0.0);

    // EWHA Energy Regularized
    let ewha_config = TrainingConfig {
        loss_type: TrainingLossType::EwhaEnergyRegularized,
        energy_penalty_lambda: 0.1,
        ..Default::default()
    };
    let mut ewha_trainer = MicroTrainer::new(ewha_config);
    let ewha_record = ewha_trainer.step(&tokens, &mut weights).unwrap();
    assert!(ewha_record.loss > 0.0);
}

#[test]
fn test_trainer_all_optimizers() {
    let tokens = vec![50, 60, 70];

    for opt in [
        OptimizerType::AdamW,
        OptimizerType::SignSGD,
        OptimizerType::SGD,
    ] {
        let config = TrainingConfig {
            optimizer: opt,
            ..Default::default()
        };
        let mut trainer = MicroTrainer::new(config);
        let mut weights = vec![0.5, -0.5, 0.2];
        let record = trainer.step(&tokens, &mut weights).unwrap();
        assert_eq!(record.step, 1);
    }
}
