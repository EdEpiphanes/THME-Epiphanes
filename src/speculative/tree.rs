use serde::{Deserialize, Serialize};

/// A node in a multi-candidate speculative draft tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeTreeNode {
    /// Unique node identifier in the tree
    pub id: usize,
    /// Parent node identifier (None for root)
    pub parent_id: Option<usize>,
    /// Tree depth level (0 is root)
    pub depth: usize,
    /// Simulated or predicted token identifier
    pub token_id: u32,
    /// Model confidence / probability score
    pub confidence: f32,
}

/// A multi-branch speculative candidate tree topology with tree-attention causal mask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeTreeTopology {
    /// List of all tree nodes
    pub nodes: Vec<SpeculativeTreeNode>,
    /// Branching depth
    pub max_depth: usize,
    /// Total candidate draft tokens in tree
    pub total_nodes: usize,
    /// Number of candidate leaf paths
    pub num_paths: usize,
    /// 2D flattened boolean tree attention mask [total_nodes x total_nodes]
    /// mask[i * total_nodes + j] == true if token j is an ancestor of token i (or j == i)
    pub tree_attention_mask: Vec<bool>,
}

impl SpeculativeTreeTopology {
    /// Builds a canonical branching candidate tree (e.g., Medusa/Eagle style)
    /// depths and branching factors, e.g. branching_factors = [3, 2, 1]
    pub fn build_canonical(branching_factors: &[usize]) -> Self {
        let mut nodes = Vec::new();
        let mut node_counter = 0;

        // Root node
        nodes.push(SpeculativeTreeNode {
            id: 0,
            parent_id: None,
            depth: 0,
            token_id: 101,
            confidence: 1.0,
        });
        node_counter += 1;

        let mut current_level_parents = vec![0];

        for (depth_idx, &branch_factor) in branching_factors.iter().enumerate() {
            let depth = depth_idx + 1;
            let mut next_level_parents = Vec::new();

            for &parent in &current_level_parents {
                for b in 0..branch_factor {
                    let node_id = node_counter;
                    node_counter += 1;

                    let conf = 0.90 / ((depth as f32) + (b as f32 * 0.15));

                    nodes.push(SpeculativeTreeNode {
                        id: node_id,
                        parent_id: Some(parent),
                        depth,
                        token_id: 1000 + (node_id as u32),
                        confidence: conf,
                    });

                    next_level_parents.push(node_id);
                }
            }

            current_level_parents = next_level_parents;
        }

        let total_nodes = nodes.len();
        let max_depth = branching_factors.len();
        let num_paths = current_level_parents.len();

        // Synthesize the 2D causal tree attention mask [total_nodes x total_nodes]
        let mut tree_mask = vec![false; total_nodes * total_nodes];

        for i in 0..total_nodes {
            // A node always attends to itself
            tree_mask[i * total_nodes + i] = true;

            // Traverse ancestors up to root
            let mut curr = nodes[i].parent_id;
            while let Some(parent) = curr {
                tree_mask[i * total_nodes + parent] = true;
                curr = nodes[parent].parent_id;
            }
        }

        Self {
            nodes,
            max_depth,
            total_nodes,
            num_paths,
            tree_attention_mask: tree_mask,
        }
    }

    /// Checks if token j is an ancestor of token i in the tree attention mask
    pub fn can_attend(&self, query_node: usize, key_node: usize) -> bool {
        if query_node < self.total_nodes && key_node < self.total_nodes {
            self.tree_attention_mask[query_node * self.total_nodes + key_node]
        } else {
            false
        }
    }

    /// Evaluates speculative tree execution: finds the longest valid accepted path
    pub fn evaluate_tree_acceptance(&self, base_fidelity: f32) -> (Vec<usize>, usize, f64) {
        // Collect all root-to-leaf paths
        let mut paths: Vec<Vec<usize>> = Vec::new();

        fn build_paths(
            curr_node: usize,
            current_path: Vec<usize>,
            nodes: &[SpeculativeTreeNode],
            paths: &mut Vec<Vec<usize>>,
        ) {
            let mut next_path = current_path;
            next_path.push(curr_node);

            let children: Vec<usize> = nodes
                .iter()
                .filter(|n| n.parent_id == Some(curr_node))
                .map(|n| n.id)
                .collect();

            if children.is_empty() {
                paths.push(next_path);
            } else {
                for child in children {
                    build_paths(child, next_path.clone(), nodes, paths);
                }
            }
        }

        build_paths(0, Vec::new(), &self.nodes, &mut paths);

        let mut best_path = Vec::new();
        let mut best_accepted_count = 0;

        for path in &paths {
            let mut accepted_count = 0;
            for (step, &node_idx) in path.iter().enumerate() {
                if step == 0 {
                    continue; // Root is already context
                }
                let node = &self.nodes[node_idx];
                let p = (node.confidence * base_fidelity).clamp(0.20, 0.98);
                let pseudo_rand = (((node_idx + 1) * 47 + step * 29) % 100) as f32 / 100.0;

                if pseudo_rand <= p {
                    accepted_count += 1;
                } else {
                    break;
                }
            }

            if accepted_count > best_accepted_count {
                best_accepted_count = accepted_count;
                best_path = path.iter().take(accepted_count + 1).cloned().collect();
            }
        }

        let tree_acceptance_rate = best_accepted_count as f64 / self.max_depth.max(1) as f64;
        (best_path, best_accepted_count, tree_acceptance_rate)
    }
}
