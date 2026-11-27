// SPDX-License-Identifier: AGPL-3.0-only
//! Architecture-neutral operation graph (v0.2.9.1, WP14).
//!
//! A model is a directed acyclic graph of [`Op`]s over 2-D tensors shaped `[rows, cols]`, where
//! rows are token positions. Shape inference runs when a node is added: each op checks its
//! input shapes and derives its output shape, so a wrong wiring fails at construction, before a
//! weight is read. The ops carry no family name. [`decoder_graph`] is one builder on top of
//! them, and [`Graph::bind_weights`] checks the graph's weight shapes against a model's tensors.

use anyhow::{bail, Context, Result};

use super::tensor::TensorIR;

/// Index of a node in its [`Graph`]. Inputs always precede the node that uses them.
pub type NodeId = usize;

/// A 2-D tensor shape: `rows` token positions by `cols` features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

/// The operations a decoder-style model is built from.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// A graph input (token ids use `cols == 1`). Added with [`Graph::input`].
    Input,
    /// Gathers one row of a `vocab x dim` table per token.
    Embedding { vocab: usize, dim: usize },
    /// Scales each row by its reciprocal root mean square and a learned vector of `dim`.
    RmsNorm { dim: usize, eps: f32 },
    /// Multiplies `in_dim` columns by an `out_dim x in_dim` weight (row-major, Hugging Face layout).
    Linear { in_dim: usize, out_dim: usize },
    /// Rotary position embedding over each head of `head_dim` values.
    Rope { head_dim: usize },
    /// Causal grouped-query attention over `q` (`num_heads` heads), `k` and `v` (`num_kv_heads`).
    Attention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    /// `silu(gate) * up`, element-wise.
    SwiGlu,
    /// Element-wise sum of two tensors of the same shape (residual connection).
    Add,
}

/// One node: an op, its inputs, the tensor that holds its weight (if any) and its shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub weight: Option<String>,
    pub shape: Shape,
}

impl Node {
    /// The shape the weight tensor must have, or `None` for ops without a weight.
    pub fn weight_shape(&self) -> Option<Vec<usize>> {
        match self.op {
            Op::Embedding { vocab, dim } => Some(vec![vocab, dim]),
            Op::RmsNorm { dim, .. } => Some(vec![dim]),
            Op::Linear { in_dim, out_dim } => Some(vec![out_dim, in_dim]),
            _ => None,
        }
    }
}

/// A directed acyclic graph of operations with inferred shapes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes
            .get(id)
            .with_context(|| format!("graph has no node {id}"))
    }

    /// Adds a graph input of the given shape.
    pub fn input(&mut self, rows: usize, cols: usize) -> NodeId {
        self.push(Op::Input, Vec::new(), None, Shape { rows, cols })
    }

    /// Adds an op, checks its inputs and infers its output shape.
    pub fn add(&mut self, op: Op, inputs: &[NodeId], weight: Option<&str>) -> Result<NodeId> {
        let shapes = inputs
            .iter()
            .map(|&id| self.node(id).map(|n| n.shape))
            .collect::<Result<Vec<_>>>()?;
        let shape = infer_shape(&op, &shapes)?;
        Ok(self.push(op, inputs.to_vec(), weight.map(str::to_string), shape))
    }

    fn push(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        weight: Option<String>,
        shape: Shape,
    ) -> NodeId {
        self.nodes.push(Node {
            op,
            inputs,
            weight,
            shape,
        });
        self.nodes.len() - 1
    }

    /// Checks every weight the graph names against the model's tensors: the tensor must exist
    /// (by exact name or dotted suffix, as the forward pass finds them) and have the shape the
    /// op needs.
    pub fn bind_weights(&self, tensors: &[TensorIR]) -> Result<()> {
        for node in &self.nodes {
            let (Some(name), Some(expected)) = (&node.weight, node.weight_shape()) else {
                continue;
            };
            let dotted = format!(".{name}");
            let tensor = tensors
                .iter()
                .find(|t| t.name == *name || t.name.ends_with(&dotted))
                .with_context(|| format!("model has no tensor for weight '{name}'"))?;
            if tensor.shape != expected {
                bail!(
                    "weight '{name}' has shape {:?}, the graph needs {expected:?}",
                    tensor.shape
                );
            }
        }
        Ok(())
    }
}

fn infer_shape(op: &Op, inputs: &[Shape]) -> Result<Shape> {
    let arity = |n: usize| -> Result<()> {
        if inputs.len() == n {
            Ok(())
        } else {
            bail!("{op:?} takes {n} input(s), got {}", inputs.len())
        }
    };
    match op {
        Op::Input => {
            arity(0)?;
            bail!("graph inputs are added with Graph::input, not Graph::add")
        }
        Op::Embedding { dim, .. } => {
            arity(1)?;
            if inputs[0].cols != 1 {
                bail!(
                    "Embedding needs token ids of shape [T, 1], got {:?}",
                    inputs[0]
                );
            }
            Ok(Shape {
                rows: inputs[0].rows,
                cols: *dim,
            })
        }
        Op::RmsNorm { dim, .. } => {
            arity(1)?;
            if inputs[0].cols != *dim {
                bail!("RmsNorm of width {dim} got {:?}", inputs[0]);
            }
            Ok(inputs[0])
        }
        Op::Linear { in_dim, out_dim } => {
            arity(1)?;
            if inputs[0].cols != *in_dim {
                bail!("Linear with in_dim {in_dim} got {:?}", inputs[0]);
            }
            Ok(Shape {
                rows: inputs[0].rows,
                cols: *out_dim,
            })
        }
        Op::Rope { head_dim } => {
            arity(1)?;
            let even = head_dim.is_multiple_of(2);
            if *head_dim == 0 || !even || !inputs[0].cols.is_multiple_of(*head_dim) {
                bail!(
                    "Rope over head_dim {head_dim} needs an even head_dim dividing {:?}",
                    inputs[0]
                );
            }
            Ok(inputs[0])
        }
        Op::Attention {
            num_heads,
            num_kv_heads,
            head_dim,
        } => {
            arity(3)?;
            if *num_kv_heads == 0 || !num_heads.is_multiple_of(*num_kv_heads) {
                bail!(
                    "{num_heads} query heads cannot be grouped over {num_kv_heads} key/value heads"
                );
            }
            let (q, k, v) = (inputs[0], inputs[1], inputs[2]);
            let q_width = num_heads * head_dim;
            let kv_width = num_kv_heads * head_dim;
            if q.cols != q_width || k.cols != kv_width || v.cols != kv_width {
                bail!(
                    "Attention expects q {q_width}, k and v {kv_width} columns, got {q:?}, {k:?}, {v:?}"
                );
            }
            if k.rows != q.rows || v.rows != q.rows {
                bail!("Attention inputs disagree on the number of positions");
            }
            Ok(Shape {
                rows: q.rows,
                cols: q_width,
            })
        }
        Op::SwiGlu => {
            arity(2)?;
            if inputs[0] != inputs[1] {
                bail!("SwiGlu gate {:?} and up {:?} differ", inputs[0], inputs[1]);
            }
            Ok(inputs[0])
        }
        Op::Add => {
            arity(2)?;
            if inputs[0] != inputs[1] {
                bail!("Add of {:?} and {:?}", inputs[0], inputs[1]);
            }
            Ok(inputs[0])
        }
    }
}

/// Hyperparameters a decoder-only, LLaMA-style graph needs.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub intermediate: usize,
    pub eps: f32,
}

/// Builds the graph of a LLaMA-style decoder over `tokens` positions, with Hugging Face tensor
/// names. Head width is `hidden / heads`.
pub fn decoder_graph(spec: &DecoderSpec, tokens: usize) -> Result<Graph> {
    if spec.heads == 0 || !spec.hidden.is_multiple_of(spec.heads) {
        bail!(
            "hidden size {} is not divisible by {} heads",
            spec.hidden,
            spec.heads
        );
    }
    let head_dim = spec.hidden / spec.heads;
    let kv_width = spec.kv_heads * head_dim;
    let h = spec.hidden;
    let mut g = Graph::new();
    let ids = g.input(tokens, 1);
    let mut x = g.add(
        Op::Embedding {
            vocab: spec.vocab,
            dim: h,
        },
        &[ids],
        Some("model.embed_tokens.weight"),
    )?;
    let norm = |g: &mut Graph, input: NodeId, weight: &str| {
        g.add(
            Op::RmsNorm {
                dim: h,
                eps: spec.eps,
            },
            &[input],
            Some(weight),
        )
    };
    let linear = |g: &mut Graph, input: NodeId, in_dim: usize, out_dim: usize, weight: &str| {
        g.add(Op::Linear { in_dim, out_dim }, &[input], Some(weight))
    };
    for i in 0..spec.layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let n1 = norm(&mut g, x, &p("input_layernorm.weight"))?;
        let q = linear(&mut g, n1, h, h, &p("self_attn.q_proj.weight"))?;
        let k = linear(&mut g, n1, h, kv_width, &p("self_attn.k_proj.weight"))?;
        let v = linear(&mut g, n1, h, kv_width, &p("self_attn.v_proj.weight"))?;
        let q = g.add(Op::Rope { head_dim }, &[q], None)?;
        let k = g.add(Op::Rope { head_dim }, &[k], None)?;
        let attn = g.add(
            Op::Attention {
                num_heads: spec.heads,
                num_kv_heads: spec.kv_heads,
                head_dim,
            },
            &[q, k, v],
            None,
        )?;
        let o = linear(&mut g, attn, h, h, &p("self_attn.o_proj.weight"))?;
        let mid = g.add(Op::Add, &[x, o], None)?;
        let n2 = norm(&mut g, mid, &p("post_attention_layernorm.weight"))?;
        let gate = linear(&mut g, n2, h, spec.intermediate, &p("mlp.gate_proj.weight"))?;
        let up = linear(&mut g, n2, h, spec.intermediate, &p("mlp.up_proj.weight"))?;
        let act = g.add(Op::SwiGlu, &[gate, up], None)?;
        let down = linear(
            &mut g,
            act,
            spec.intermediate,
            h,
            &p("mlp.down_proj.weight"),
        )?;
        x = g.add(Op::Add, &[mid, down], None)?;
    }
    let normed = norm(&mut g, x, "model.norm.weight")?;
    linear(&mut g, normed, h, spec.vocab, "lm_head.weight")?;
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-worked: 16 hidden, 4 query heads of 4 over 2 key/value heads, FFN 32, vocab 40.
    fn tiny_spec(layers: usize) -> DecoderSpec {
        DecoderSpec {
            vocab: 40,
            hidden: 16,
            layers,
            heads: 4,
            kv_heads: 2,
            intermediate: 32,
            eps: 1e-5,
        }
    }

    fn shape(rows: usize, cols: usize) -> Shape {
        Shape { rows, cols }
    }

    fn named<'a>(g: &'a Graph, weight: &str) -> &'a Node {
        g.nodes()
            .iter()
            .find(|n| n.weight.as_deref() == Some(weight))
            .unwrap()
    }

    #[test]
    fn decoder_graph_has_hand_counted_nodes_and_shapes() {
        // 5 tokens. Nodes: input, embedding, 15 per layer (norm, q/k/v, 2 rope, attention,
        // o, residual add, norm, gate/up, swiglu, down, residual add), final norm, head.
        let g = decoder_graph(&tiny_spec(2), 5).unwrap();
        assert_eq!(g.nodes().len(), 2 + 2 * 15 + 2);
        assert_eq!(g.nodes()[0].shape, shape(5, 1));
        assert_eq!(g.nodes()[1].shape, shape(5, 16));
        assert_eq!(g.nodes().last().unwrap().shape, shape(5, 40));
        // Queries keep 16 columns; keys and values have 2 heads of 4 = 8 columns; FFN is 32.
        assert_eq!(
            named(&g, "model.layers.0.self_attn.q_proj.weight").shape,
            shape(5, 16)
        );
        assert_eq!(
            named(&g, "model.layers.1.self_attn.k_proj.weight").shape,
            shape(5, 8)
        );
        assert_eq!(
            named(&g, "model.layers.1.mlp.gate_proj.weight").shape,
            shape(5, 32)
        );
        assert_eq!(
            named(&g, "model.layers.1.mlp.down_proj.weight").shape,
            shape(5, 16)
        );
    }

    #[test]
    fn weight_shapes_are_out_by_in() {
        let g = decoder_graph(&tiny_spec(1), 3).unwrap();
        let w = |name: &str| named(&g, name).weight_shape();
        assert_eq!(
            w("model.layers.0.self_attn.k_proj.weight"),
            Some(vec![8, 16])
        );
        assert_eq!(w("model.layers.0.mlp.down_proj.weight"), Some(vec![16, 32]));
        assert_eq!(w("model.embed_tokens.weight"), Some(vec![40, 16]));
        assert_eq!(w("model.layers.0.input_layernorm.weight"), Some(vec![16]));
    }

    #[test]
    fn attention_rejects_query_heads_not_grouped_over_kv_heads() {
        let mut g = Graph::new();
        let q = g.input(5, 12);
        let k = g.input(5, 8);
        let v = g.input(5, 8);
        // 3 query heads of 4 cannot be grouped over 2 key/value heads.
        let bad = Op::Attention {
            num_heads: 3,
            num_kv_heads: 2,
            head_dim: 4,
        };
        assert!(g.add(bad, &[q, k, v], None).is_err());
    }

    #[test]
    fn attention_output_is_query_width_and_rejects_wrong_kv_width() {
        let mut g = Graph::new();
        let q = g.input(5, 16);
        let k = g.input(5, 16); // should be 8 for 2 kv heads of 4
        let v = g.input(5, 8);
        let op = Op::Attention {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
        };
        assert!(g.add(op.clone(), &[q, k, v], None).is_err());
        let k = g.input(5, 8);
        let id = g.add(op, &[q, k, v], None).unwrap();
        assert_eq!(g.node(id).unwrap().shape, shape(5, 16));
    }

    #[test]
    fn residual_add_rejects_different_shapes() {
        let mut g = Graph::new();
        let a = g.input(5, 16);
        let b = g.input(5, 8);
        assert!(g.add(Op::Add, &[a, b], None).is_err());
        assert!(g.add(Op::Add, &[a], None).is_err());
    }

    #[test]
    fn linear_rejects_an_input_width_it_was_not_built_for() {
        let mut g = Graph::new();
        let x = g.input(5, 16);
        let op = Op::Linear {
            in_dim: 8,
            out_dim: 4,
        };
        assert!(g.add(op, &[x], Some("w")).is_err());
    }

    #[test]
    fn rope_needs_an_even_head_width_that_divides_the_columns() {
        let mut g = Graph::new();
        let x = g.input(5, 12);
        assert!(g.add(Op::Rope { head_dim: 3 }, &[x], None).is_err());
        assert!(g.add(Op::Rope { head_dim: 8 }, &[x], None).is_err());
        assert!(g.add(Op::Rope { head_dim: 4 }, &[x], None).is_ok());
    }

    #[test]
    fn inputs_must_already_exist() {
        let mut g = Graph::new();
        let a = g.input(5, 16);
        assert!(g.add(Op::Add, &[a, 7], None).is_err());
    }

    #[test]
    fn embedding_needs_one_column_of_token_ids() {
        let mut g = Graph::new();
        let ids = g.input(5, 2);
        let op = Op::Embedding { vocab: 40, dim: 16 };
        assert!(g
            .add(op, &[ids], Some("model.embed_tokens.weight"))
            .is_err());
    }

    /// Hand-written tensor list for one layer, names and shapes from the Hugging Face layout.
    fn one_layer_tensors(down_shape: Vec<usize>) -> Vec<TensorIR> {
        let t = |name: &str, shape: Vec<usize>| TensorIR::new(name.into(), shape, "F32".into(), 0);
        vec![
            t("model.embed_tokens.weight", vec![40, 16]),
            t("model.layers.0.input_layernorm.weight", vec![16]),
            t("model.layers.0.self_attn.q_proj.weight", vec![16, 16]),
            t("model.layers.0.self_attn.k_proj.weight", vec![8, 16]),
            t("model.layers.0.self_attn.v_proj.weight", vec![8, 16]),
            t("model.layers.0.self_attn.o_proj.weight", vec![16, 16]),
            t("model.layers.0.post_attention_layernorm.weight", vec![16]),
            t("model.layers.0.mlp.gate_proj.weight", vec![32, 16]),
            t("model.layers.0.mlp.up_proj.weight", vec![32, 16]),
            t("model.layers.0.mlp.down_proj.weight", down_shape),
            t("model.norm.weight", vec![16]),
            t("lm_head.weight", vec![40, 16]),
        ]
    }

    #[test]
    fn bind_weights_accepts_matching_tensors() {
        let g = decoder_graph(&tiny_spec(1), 3).unwrap();
        g.bind_weights(&one_layer_tensors(vec![16, 32])).unwrap();
    }

    #[test]
    fn bind_weights_names_a_transposed_or_missing_tensor() {
        let g = decoder_graph(&tiny_spec(1), 3).unwrap();
        let err = g
            .bind_weights(&one_layer_tensors(vec![32, 16]))
            .unwrap_err();
        assert!(format!("{err:#}").contains("model.layers.0.mlp.down_proj.weight"));
        let mut tensors = one_layer_tensors(vec![16, 32]);
        tensors.retain(|t| !t.name.ends_with("up_proj.weight"));
        let err = g.bind_weights(&tensors).unwrap_err();
        assert!(format!("{err:#}").contains("up_proj.weight"));
    }
}
