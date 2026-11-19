// SPDX-License-Identifier: AGPL-3.0-only
use serde_json::{json, Value};
use std::path::PathBuf;
use thme::cli::mcp::process_json_rpc;

fn resolve_target_model() -> Option<PathBuf> {
    // Tests that open a real model file are opt-in: set THME_REAL_MODEL_TESTS=1 to run them.
    if std::env::var_os("THME_REAL_MODEL_TESTS").is_none() {
        return None;
    }
    if let Ok(path_str) = std::env::var("THME_MEDGEMMA_PATH") {
        let p = PathBuf::from(path_str);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        let p = PathBuf::from(local_app_data).join("hermes/models/medgemma-4b-it-UD-Q4_K_XL.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    let rel = PathBuf::from("models/medgemma-4b-it-UD-Q4_K_XL.gguf");
    if rel.exists() {
        Some(rel)
    } else {
        None
    }
}

#[test]
fn test_mcp_initialize_and_tools_list() {
    let mut model_path = None;

    // 1. Handshake (initialize)
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let init_res_str =
        process_json_rpc(&init_req.to_string(), &mut model_path).expect("response to initialize");
    let init_res: Value = serde_json::from_str(&init_res_str).expect("parse initialize json");
    assert_eq!(init_res["jsonrpc"], "2.0");
    assert_eq!(init_res["id"], 1);
    assert_eq!(init_res["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(init_res["result"]["serverInfo"]["name"], "thme-mcp");

    // 2. Tools list
    let list_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });
    let list_res_str =
        process_json_rpc(&list_req.to_string(), &mut model_path).expect("response to tools/list");
    let list_res: Value = serde_json::from_str(&list_res_str).expect("parse tools/list json");
    assert_eq!(list_res["id"], 2);
    let tools = list_res["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 11);

    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(tool_names.contains(&"thme_inspect_model"));
    assert!(tool_names.contains(&"thme_auto_decide"));
    assert!(tool_names.contains(&"thme_apply_surgery"));
    assert!(tool_names.contains(&"thme_validate_quality"));
    assert!(tool_names.contains(&"thme_profile_roofline"));
    assert!(tool_names.contains(&"thme_forward_pass"));
}

#[test]
fn test_mcp_error_handling() {
    let mut model_path = None;

    // Malformed JSON -> -32700
    let err_str = process_json_rpc("{malformed json", &mut model_path).expect("parse error");
    let err_json: Value = serde_json::from_str(&err_str).expect("parse err json");
    assert_eq!(err_json["error"]["code"], -32700);

    // Unknown method -> -32601
    let unk_req = json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "non_existent_method",
        "params": {}
    });
    let unk_str =
        process_json_rpc(&unk_req.to_string(), &mut model_path).expect("method not found");
    let unk_json: Value = serde_json::from_str(&unk_str).expect("parse unk json");
    assert_eq!(unk_json["error"]["code"], -32601);
}

#[test]
fn test_mcp_validate_quality_format_bug_fix() {
    let mut model_path = None;
    let call_req = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "thme_validate_quality",
            "arguments": {
                "baseline_ppl": 10.5,
                "candidate_ppl": 10.6,
                "max_delta_ppl": 0.25,
                "corpus": "wikitext2"
            }
        }
    });

    let res_str = process_json_rpc(&call_req.to_string(), &mut model_path).expect("validate call");
    let res: Value = serde_json::from_str(&res_str).expect("parse call json");
    let text = res["result"]["content"][0]["text"]
        .as_str()
        .expect("text content");

    // Assert that - **Corpus**: is followed by "wikitext2" and NOT the numeric baseline PPL (10.5)
    assert!(
        text.contains("- **Corpus**: wikitext2"),
        "Output did not contain corrected corpus slot: {}",
        text
    );
    assert!(text.contains("- **Baseline PPL**: 10.5000"));
    assert!(text.contains("- **Candidate PPL**: 10.6000"));
    assert!(text.contains("- **Gate Verdict**: **Accept**"));
}

#[test]
fn test_mcp_profile_roofline() {
    let mut model_path = None;
    let call_req = json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "thme_profile_roofline",
            "arguments": {
                "hardware": "rtx4090"
            }
        }
    });

    let res_str = process_json_rpc(&call_req.to_string(), &mut model_path).expect("roofline call");
    let res: Value = serde_json::from_str(&res_str).expect("parse call json");
    let text = res["result"]["content"][0]["text"]
        .as_str()
        .expect("text content");
    assert!(text.contains("NVIDIA GeForce RTX 4090"));
    assert!(text.contains("FlashAttention-2"));
}

#[test]
fn test_mcp_real_model_lifecycle() {
    let target = match resolve_target_model() {
        Some(p) => p,
        None => {
            eprintln!("Target MedGemma model not found on disk; skipping live model MCP test");
            return;
        }
    };

    let target_str = target.to_str().expect("valid target str").to_string();
    let mut model_path = Some(target_str.clone());

    // 1. Inspect model
    let inspect_req = json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "thme_inspect_model",
            "arguments": {
                "model_path": target_str
            }
        }
    });
    let inspect_str = process_json_rpc(&inspect_req.to_string(), &mut model_path).expect("inspect");
    let inspect_res: Value = serde_json::from_str(&inspect_str).expect("parse inspect json");
    let text = inspect_res["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    assert!(text.contains("Gemma-family"));
    assert!(text.contains("34"));
    assert!(text.contains("2560"));

    // 2. Auto decide on real model tensors
    let decide_req = json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "thme_auto_decide",
            "arguments": {
                "quality_tolerance": 0.02
            }
        }
    });
    let decide_str = process_json_rpc(&decide_req.to_string(), &mut model_path).expect("decide");
    let decide_res: Value = serde_json::from_str(&decide_str).expect("parse decide json");
    let text = decide_res["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    assert!(text.contains("34 layers"));

    // 3. Apply surgery on real model
    let surgery_req = json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "thme_apply_surgery",
            "arguments": {
                "recipe_json": r#"{
                    "name": "medgemma_q4_surgery",
                    "author": "CodeGPT",
                    "target_model": "medgemma",
                    "version": "1.0",
                    "description": "Set precision for layer 0",
                    "operations": [
                        {
                            "type": "SetPrecision",
                            "params": {
                                "start_layer": 0,
                                "end_layer": 1,
                                "format": "FP16"
                            }
                        }
                    ]
                }"#
            }
        }
    });
    let surgery_str = process_json_rpc(&surgery_req.to_string(), &mut model_path).expect("surgery");
    let surgery_res: Value = serde_json::from_str(&surgery_str).expect("parse surgery json");
    let text = surgery_res["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    assert!(text.contains("Surgery Recipe Executed Successfully"));
    assert!(text.contains("34 layers"));
}

#[test]
fn test_mcp_forward_pass_tool() {
    let mut model_path = None;

    // 1. Missing tokens error
    let req = json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "thme_forward_pass",
            "arguments": {}
        }
    });
    let res_str = process_json_rpc(&req.to_string(), &mut model_path).expect("call");
    let res: Value = serde_json::from_str(&res_str).expect("json");
    let text = res["result"]["content"][0]["text"].as_str().expect("text");
    assert!(text.contains("Tool execution error"));

    // 2. Tokens < 2 error
    let req = json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "thme_forward_pass",
            "arguments": {
                "tokens": [42]
            }
        }
    });
    let res_str = process_json_rpc(&req.to_string(), &mut model_path).expect("call");
    let res: Value = serde_json::from_str(&res_str).expect("json");
    let text = res["result"]["content"][0]["text"].as_str().expect("text");
    assert!(text.contains("Tool execution error"));
}
