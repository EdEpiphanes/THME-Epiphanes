// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use serde_json::{json, Value};

type TaskEdit = Box<dyn FnOnce(&mut Value)>;
type RecordEdit = Box<dyn FnOnce(&mut ResultRecord)>;

const DATA: &[u8] = b"{\"id\":\"q1\",\"input\":\"2+2\",\"target\":\"4\"}\n";

fn task_json() -> Value {
    json!({
        "schema": TASK_SCHEMA,
        "id": "toy-arith",
        "version": 1,
        "family": "checkable-answer",
        "dataset": {
            "path": "data/toy.jsonl",
            "sha256": compute_sha256(DATA),
            "split": "test",
            "licence": "CC0-1.0"
        },
        "prompt_template": "Q: {input}\nA:",
        "extract": { "kind": "after-marker", "marker": "####" },
        "metric": { "kind": "numeric-tolerance", "abs": 0.0 }
    })
}

fn task() -> TaskSpec {
    let t: TaskSpec = serde_json::from_value(task_json()).unwrap();
    t.validate().unwrap();
    t
}

fn record_for(t: &TaskSpec) -> ResultRecord {
    ResultRecord {
        schema: RECORD_SCHEMA.into(),
        benchmark: BenchmarkRef {
            id: t.id.clone(),
            version: t.version,
            task_sha256: t.hash(),
            dataset_sha256: t.dataset.sha256.clone(),
            split: t.dataset.split.clone(),
            licence: t.dataset.licence.clone(),
        },
        model: ModelRef {
            sha256: "ab".repeat(32),
            transformation: Some("recipe:q8".into()),
        },
        harness: Harness {
            version: "0.2.0".into(),
            run_mode: RunMode::Native,
            prompt_template_sha256: t.prompt_template_hash(),
        },
        decoding: Decoding {
            temperature: 0.0,
            seed: None,
            max_new_tokens: Some(16),
        },
        items: vec![
            ItemResult {
                id: "q1".into(),
                output: Some("#### 4".into()),
                score: 1.0,
            },
            ItemResult {
                id: "q2".into(),
                output: Some("#### 5".into()),
                score: 0.0,
            },
        ],
        score: 0.5,
        interval: Interval {
            low: 0.1,
            high: 0.9,
            method: IntervalMethod::Wilson,
            level: 0.95,
        },
        evidence: Evidence::Simulated,
        hardware: "synthetic test".into(),
        run_date: "2026-11-13".into(),
        duration_s: 0.0,
    }
}

fn task_err(edit: impl FnOnce(&mut Value)) -> String {
    let mut v = task_json();
    edit(&mut v);
    match serde_json::from_value::<TaskSpec>(v) {
        Err(e) => e.to_string(),
        Ok(t) => format!("{:#}", t.validate().unwrap_err()),
    }
}

fn record_err(edit: impl FnOnce(&mut ResultRecord)) -> String {
    let mut r = record_for(&task());
    edit(&mut r);
    format!("{:#}", r.validate().unwrap_err())
}

#[test]
fn valid_task_and_record_round_trip() {
    let t = task();
    let back: TaskSpec = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.hash(), t.hash());
    assert_eq!(t.key(), "toy-arith@v1");

    let r = record_for(&t);
    r.validate().unwrap();
    r.check_against(&t).unwrap();
    let back: ResultRecord = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back, r);
}

#[test]
fn malformed_tasks_are_refused() {
    let cases: Vec<(TaskEdit, &str)> = vec![
        (Box::new(|v| v["schema"] = json!("other/v1")), "schema"),
        (Box::new(|v| v["id"] = json!("Bad Id")), "id"),
        (Box::new(|v| v["version"] = json!(0)), "version"),
        (Box::new(|v| v["extra"] = json!(1)), "unknown field"),
        (
            Box::new(|v| {
                v.as_object_mut().unwrap().remove("metric");
            }),
            "missing field",
        ),
        (
            Box::new(|v| v["dataset"]["sha256"] = json!("ABC")),
            "sha256",
        ),
        (
            Box::new(|v| v["dataset"]["path"] = json!("/abs/toy.jsonl")),
            "relative",
        ),
        (
            Box::new(|v| v["dataset"]["path"] = json!("../toy.jsonl")),
            "relative",
        ),
        (
            Box::new(|v| v["dataset"]["path"] = json!(r"data\toy.jsonl")),
            "relative",
        ),
        (
            Box::new(|v| v["dataset"]["licence"] = json!(" ")),
            "licence",
        ),
        (
            Box::new(|v| v["prompt_template"] = json!("no placeholder")),
            "{input}",
        ),
        (
            Box::new(|v| v["prompt_template"] = json!("{input} {choices}")),
            "multiple-choice",
        ),
        (
            Box::new(|v| v["extract"] = json!({"kind": "after-marker", "marker": ""})),
            "marker",
        ),
        (
            Box::new(|v| v["metric"] = json!({"kind": "numeric-tolerance"})),
            "abs, rel",
        ),
        (
            Box::new(|v| v["metric"] = json!({"kind": "numeric-tolerance", "abs": -1.0})),
            "invalid tolerance",
        ),
        (
            Box::new(|v| v["metric"] = json!({"kind": "pass-at-k", "k": 5, "samples": 2})),
            "k <= samples",
        ),
        (
            Box::new(|v| v["metric"] = json!({"kind": "choice-by-likelihood"})),
            "cannot use",
        ),
        (
            Box::new(|v| v["extract"] = json!({"kind": "none"})),
            "cannot use",
        ),
        (Box::new(|v| v["family"] = json!("code")), "cannot use"),
    ];
    for (edit, expected) in cases {
        let err = task_err(edit);
        assert!(err.contains(expected), "expected '{expected}' in: {err}");
    }
}

#[test]
fn family_metric_pairs_that_work() {
    let mut v = task_json();
    v["family"] = json!("multiple-choice");
    v["prompt_template"] = json!("{input}\n{choices}\nAnswer:");
    v["extract"] = json!({"kind": "none"});
    v["metric"] = json!({"kind": "choice-by-likelihood"});
    serde_json::from_value::<TaskSpec>(v.clone())
        .unwrap()
        .validate()
        .unwrap();

    v["extract"] = json!({"kind": "choice-letter"});
    v["metric"] = json!({"kind": "exact-match"});
    serde_json::from_value::<TaskSpec>(v.clone())
        .unwrap()
        .validate()
        .unwrap();

    v["family"] = json!("code");
    v["prompt_template"] = json!("{input}");
    v["extract"] = json!({"kind": "none"});
    v["metric"] = json!({"kind": "pass-at-k", "k": 1, "samples": 5});
    serde_json::from_value::<TaskSpec>(v)
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn malformed_records_are_refused() {
    let cases: Vec<(RecordEdit, &str)> = vec![
        (Box::new(|r| r.schema = "x".into()), "schema"),
        (
            Box::new(|r| r.model.sha256 = "short".into()),
            "model.sha256",
        ),
        (Box::new(|r| r.hardware = String::new()), "hardware"),
        (
            Box::new(|r| r.evidence = Evidence::Imported),
            "needs run_mode",
        ),
        (
            Box::new(|r| {
                r.harness.run_mode = RunMode::Imported;
                r.evidence = Evidence::Measured;
            }),
            "labelled imported",
        ),
        (Box::new(|r| r.decoding.temperature = 0.7), "needs a seed"),
        (
            Box::new(|r| r.decoding.temperature = f64::NAN),
            "temperature",
        ),
        (Box::new(|r| r.items.clear()), "per-item"),
        (Box::new(|r| r.items[1].id = "q1".into()), "twice"),
        (Box::new(|r| r.items[0].score = 1.5), "outside [0, 1]"),
        (Box::new(|r| r.score = 0.6), "mean"),
        (Box::new(|r| r.interval.low = 0.6), "does not contain"),
        (Box::new(|r| r.interval.high = 1.2), "outside [0, 1]"),
        (Box::new(|r| r.interval.level = 95.0), "level"),
        (Box::new(|r| r.duration_s = -1.0), "duration"),
        (Box::new(|r| r.run_date = "13/11/2026".into()), "YYYY-MM-DD"),
        (Box::new(|r| r.run_date = "2026-13-01".into()), "YYYY-MM-DD"),
    ];
    for (edit, expected) in cases {
        let err = record_err(edit);
        assert!(err.contains(expected), "expected '{expected}' in: {err}");
    }

    // Unknown fields in a record file are refused too.
    let mut v = serde_json::to_value(record_for(&task())).unwrap();
    v["benchmark"]["note"] = json!("x");
    assert!(serde_json::from_value::<ResultRecord>(v)
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
}

#[test]
fn sampled_imported_record_is_accepted() {
    let mut r = record_for(&task());
    r.decoding.temperature = 0.7;
    r.decoding.seed = Some(7);
    r.harness.run_mode = RunMode::Imported;
    r.evidence = Evidence::Imported;
    r.validate().unwrap();
}

#[test]
fn record_must_match_its_task() {
    let t = task();
    let r = record_for(&t);
    let mut other = t.clone();
    other.prompt_template = "Question: {input}\nAnswer:".into();
    let err = format!("{:#}", r.check_against(&other).unwrap_err());
    assert!(err.contains("task hash"), "{err}");

    let mut r2 = r.clone();
    r2.harness.prompt_template_sha256 = "cd".repeat(32);
    let err = format!("{:#}", r2.check_against(&t).unwrap_err());
    assert!(err.contains("prompt template hash"), "{err}");

    let mut r3 = r;
    r3.benchmark.version = 2;
    assert!(format!("{:#}", r3.check_against(&t).unwrap_err()).contains("version"));
}

#[test]
fn registry_loads_folder_and_refuses_duplicates_and_bad_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("data")).unwrap();
    std::fs::write(dir.path().join("data/toy.jsonl"), DATA).unwrap();
    std::fs::write(dir.path().join("a.json"), task_json().to_string()).unwrap();
    let mut v2 = task_json();
    v2["version"] = json!(2);
    std::fs::write(dir.path().join("b.json"), v2.to_string()).unwrap();
    std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();

    let reg = Registry::load(dir.path()).unwrap();
    assert_eq!(reg.len(), 2);
    let keys: Vec<String> = reg.iter().map(|(_, t)| t.key()).collect();
    assert_eq!(keys, ["toy-arith@v1", "toy-arith@v2"]);
    let t = reg.get("toy-arith", 1).unwrap();
    assert!(reg.get("toy-arith", 3).is_none());

    // Dataset hash check.
    t.verify_dataset(dir.path()).unwrap();
    std::fs::write(dir.path().join("data/toy.jsonl"), b"changed").unwrap();
    let err = format!("{:#}", t.verify_dataset(dir.path()).unwrap_err());
    assert!(err.contains("expects"), "{err}");

    // Duplicate id@version.
    std::fs::write(dir.path().join("c.json"), task_json().to_string()).unwrap();
    let err = format!("{:#}", Registry::load(dir.path()).unwrap_err());
    assert!(err.contains("defined twice"), "{err}");
    std::fs::remove_file(dir.path().join("c.json")).unwrap();

    // One malformed file refuses the folder.
    std::fs::write(dir.path().join("d.json"), "{\"schema\": 1}").unwrap();
    let err = format!("{:#}", Registry::load(dir.path()).unwrap_err());
    assert!(err.contains("d.json"), "{err}");
}
