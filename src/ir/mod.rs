// SPDX-License-Identifier: AGPL-3.0-only
pub mod layer_decomp;
pub mod model;
pub mod paradigm;
pub mod rebuilder;
pub mod tensor;

pub use layer_decomp::{
    DeconstructedTensor, LayerDeconstruction, LayerSubBlock, LayerTopologyStyle, SubBlockCategory,
};
pub use model::{ModelArchitecture, ModelIR};
pub use paradigm::{
    ParadigmConversionConfig, ParadigmConversionReport, ParadigmConverter, SpecializedModelParadigm,
};
pub use rebuilder::{ArchitectureBlueprint, LayerBlueprint, ModelRebuilder, RemapStrategy};
pub use tensor::{extract_layer_index, TensorFormat, TensorIR, TensorRole, WeightProfile};
