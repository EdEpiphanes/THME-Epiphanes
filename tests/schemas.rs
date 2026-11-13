// SPDX-License-Identifier: AGPL-3.0-only
//! Versioned schemas under docs/schemas/ (THME-3009). Every schema carries its version in its
//! `$id` and file name and validates a committed example; the example also loads through the
//! Rust type the schema describes, and broken copies are refused by both. The validator below
//! covers the JSON Schema keywords these files use and fails on any other keyword, so a schema
//! cannot silently go unchecked.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// One pattern atom: character ranges, minimum and maximum repeats.
type Atom = (Vec<(char, char)>, usize, usize);
/// An edit that breaks a valid document.
type Edit = Box<dyn Fn(&mut Value)>;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load(rel: &str) -> Value {
    let text = std::fs::read_to_string(root().join(rel)).unwrap();
    serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap()
}

/// Anchored patterns made of literals and `[...]` classes, each optionally followed by `*`,
/// `+` or `{n}`.
fn pattern_matches(pattern: &str, text: &str) -> bool {
    let p = pattern
        .strip_prefix('^')
        .and_then(|p| p.strip_suffix('$'))
        .unwrap_or_else(|| panic!("pattern {pattern} must be anchored with ^ and $"));
    let mut atoms: Vec<Atom> = Vec::new();
    let chars: Vec<char> = p.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let mut set = Vec::new();
        if chars[i] == '[' {
            i += 1;
            while chars[i] != ']' {
                if chars.get(i + 1) == Some(&'-') && chars.get(i + 2).is_some_and(|c| *c != ']') {
                    set.push((chars[i], chars[i + 2]));
                    i += 3;
                } else {
                    set.push((chars[i], chars[i]));
                    i += 1;
                }
            }
            i += 1;
        } else {
            assert!(
                !"\\.()|?".contains(chars[i]),
                "unsupported pattern syntax in {pattern}"
            );
            set.push((chars[i], chars[i]));
            i += 1;
        }
        let (min, max) = match chars.get(i) {
            Some('*') => (i += 1, (0, usize::MAX)).1,
            Some('+') => (i += 1, (1, usize::MAX)).1,
            Some('{') => {
                let end = chars[i..].iter().position(|c| *c == '}').unwrap() + i;
                let n: usize = chars[i + 1..end]
                    .iter()
                    .collect::<String>()
                    .parse()
                    .unwrap();
                i = end + 1;
                (n, n)
            }
            _ => (1, 1),
        };
        atoms.push((set, min, max));
    }
    fn go(atoms: &[Atom], text: &[char]) -> bool {
        let Some(((set, min, max), rest)) = atoms.split_first() else {
            return text.is_empty();
        };
        let fits = |c: &char| set.iter().any(|(a, b)| a <= c && c <= b);
        let run = text.iter().take_while(|c| fits(c)).count().min(*max);
        (*min..=run).rev().any(|n| go(rest, &text[n..]))
    }
    let text: Vec<char> = text.chars().collect();
    go(&atoms, &text)
}

fn type_ok(t: &str, v: &Value) -> bool {
    match t {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => v.as_f64().is_some_and(|x| x.fract() == 0.0),
        other => panic!("unknown type {other}"),
    }
}

/// Errors from validating `v` against `s`; `top` resolves `$ref`.
fn validate(s: &Value, v: &Value, top: &Value, at: &str, errors: &mut Vec<String>) {
    let obj = s.as_object().unwrap();
    for (key, rule) in obj {
        match key.as_str() {
            "$schema" | "$id" | "title" | "description" | "$defs" => {}
            "$ref" => {
                let path = rule.as_str().unwrap().strip_prefix("#/").unwrap();
                let target = path.split('/').fold(top, |node, part| &node[part]);
                validate(target, v, top, at, errors);
            }
            "type" => {
                let ok = match rule {
                    Value::String(t) => type_ok(t, v),
                    Value::Array(ts) => ts.iter().any(|t| type_ok(t.as_str().unwrap(), v)),
                    _ => panic!("bad type"),
                };
                if !ok {
                    errors.push(format!("{at}: not of type {rule}"));
                }
            }
            "const" if v != rule => errors.push(format!("{at}: not {rule}")),
            "enum" if !rule.as_array().unwrap().contains(v) => {
                errors.push(format!("{at}: {v} not in {rule}"))
            }
            "const" | "enum" => {}
            "required" => {
                if let Some(o) = v.as_object() {
                    for r in rule.as_array().unwrap() {
                        if !o.contains_key(r.as_str().unwrap()) {
                            errors.push(format!("{at}: missing {r}"));
                        }
                    }
                }
            }
            "properties" => {
                if let Some(o) = v.as_object() {
                    for (k, sub) in rule.as_object().unwrap() {
                        if let Some(x) = o.get(k) {
                            validate(sub, x, top, &format!("{at}.{k}"), errors);
                        }
                    }
                }
            }
            "additionalProperties" => {
                assert_eq!(
                    rule,
                    &json!(false),
                    "only additionalProperties: false is supported"
                );
                if let Some(o) = v.as_object() {
                    let known = obj.get("properties").and_then(Value::as_object);
                    for k in o.keys() {
                        if !known.is_some_and(|p| p.contains_key(k)) {
                            errors.push(format!("{at}: unknown field {k}"));
                        }
                    }
                }
            }
            "items" => {
                if let Some(a) = v.as_array() {
                    for (i, x) in a.iter().enumerate() {
                        validate(rule, x, top, &format!("{at}[{i}]"), errors);
                    }
                }
            }
            "minItems" => {
                if v.as_array()
                    .is_some_and(|a| (a.len() as u64) < rule.as_u64().unwrap())
                {
                    errors.push(format!("{at}: too few items"));
                }
            }
            "minLength" | "maxLength" => {
                if let Some(t) = v.as_str() {
                    let n = t.chars().count() as u64;
                    let lim = rule.as_u64().unwrap();
                    if (key == "minLength" && n < lim) || (key == "maxLength" && n > lim) {
                        errors.push(format!("{at}: length {n} breaks {key} {lim}"));
                    }
                }
            }
            "pattern" => {
                if let Some(t) = v.as_str() {
                    if !pattern_matches(rule.as_str().unwrap(), t) {
                        errors.push(format!("{at}: '{t}' does not match {rule}"));
                    }
                }
            }
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
                if let Some(x) = v.as_f64() {
                    let lim = rule.as_f64().unwrap();
                    let bad = match key.as_str() {
                        "minimum" => x < lim,
                        "maximum" => x > lim,
                        "exclusiveMinimum" => x <= lim,
                        _ => x >= lim,
                    };
                    if bad {
                        errors.push(format!("{at}: {x} breaks {key} {lim}"));
                    }
                }
            }
            "oneOf" | "anyOf" => {
                let passing = rule
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|sub| {
                        let mut e = Vec::new();
                        validate(sub, v, top, at, &mut e);
                        e.is_empty()
                    })
                    .count();
                let ok = if key == "oneOf" {
                    passing == 1
                } else {
                    passing >= 1
                };
                if !ok {
                    errors.push(format!("{at}: {passing} branches of {key} match"));
                }
            }
            other => panic!("schema keyword '{other}' is not supported by this validator"),
        }
    }
}

fn errors(schema: &Value, doc: &Value) -> Vec<String> {
    let mut e = Vec::new();
    validate(schema, doc, schema, "$", &mut e);
    e
}

fn write_tmp(dir: &Path, name: &str, v: &Value) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, serde_json::to_string_pretty(v).unwrap()).unwrap();
    p
}

#[test]
fn every_schema_is_versioned_and_has_an_example() {
    let dir = root().join("docs/schemas");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    assert!(names.len() >= 4, "{names:?}");
    for name in names {
        let schema = load(&format!("docs/schemas/{name}"));
        let stem = name.trim_end_matches(".json");
        let version = stem.rsplit('-').next().unwrap();
        assert!(
            version.starts_with('v') && version[1..].parse::<u32>().is_ok(),
            "{name}: no -vN suffix"
        );
        assert!(
            schema["$id"].as_str().unwrap().ends_with(&name),
            "{name}: $id does not name the file"
        );
        assert!(schema["title"].is_string(), "{name}: no title");
    }
}

#[test]
fn benchmark_task_example_and_broken_copies() {
    let schema = load("docs/schemas/bench-task-v1.json");
    let good = load("docs/schemas/examples/bench-task-v1.json");
    assert_eq!(errors(&schema, &good), Vec::<String>::new());
    let tmp = tempfile::tempdir().unwrap();
    thme::bench::TaskSpec::load(&write_tmp(tmp.path(), "t.json", &good)).unwrap();

    let broken: Vec<Edit> = vec![
        Box::new(|v| v["extra"] = json!(1)),
        Box::new(|v| v["dataset"]["sha256"] = json!("XYZ")),
        Box::new(|v| v["id"] = json!("Bad Id")),
        Box::new(|v| v["version"] = json!(0)),
        Box::new(|v| v["metric"] = json!({"kind": "numeric-tolerance"})),
        Box::new(|v| v["metric"] = json!({"kind": "pass-at-k", "k": 0, "samples": 2})),
        Box::new(|v| v["extract"] = json!({"kind": "after-marker"})),
        Box::new(|v| v["family"] = json!("essay")),
        Box::new(|v| {
            v.as_object_mut().unwrap().remove("prompt_template");
        }),
    ];
    for (i, edit) in broken.iter().enumerate() {
        let mut v = good.clone();
        edit(&mut v);
        assert!(
            !errors(&schema, &v).is_empty(),
            "schema accepted broken task {i}"
        );
        let p = write_tmp(tmp.path(), &format!("b{i}.json"), &v);
        assert!(
            thme::bench::TaskSpec::load(&p).is_err(),
            "loader accepted broken task {i}"
        );
    }
}

#[test]
fn result_record_example_and_broken_copies() {
    let schema = load("docs/schemas/bench-record-v1.json");
    let good = load("docs/schemas/examples/bench-record-v1.json");
    assert_eq!(errors(&schema, &good), Vec::<String>::new());
    let tmp = tempfile::tempdir().unwrap();
    thme::bench::ResultRecord::load(&write_tmp(tmp.path(), "r.json", &good)).unwrap();

    let broken: Vec<Edit> = vec![
        Box::new(|v| v["items"] = json!([])),
        Box::new(|v| v["items"][0]["score"] = json!(1.5)),
        Box::new(|v| v["interval"]["method"] = json!("normal")),
        Box::new(|v| v["interval"]["level"] = json!(1.0)),
        Box::new(|v| v["evidence"] = json!("guessed")),
        Box::new(|v| v["harness"]["run_mode"] = json!("remote")),
        Box::new(|v| v["run_date"] = json!("13/11/2026")),
        Box::new(|v| v["model"]["sha256"] = json!("1".repeat(63))),
        Box::new(|v| v["benchmark"]["note"] = json!("x")),
    ];
    for (i, edit) in broken.iter().enumerate() {
        let mut v = good.clone();
        edit(&mut v);
        assert!(
            !errors(&schema, &v).is_empty(),
            "schema accepted broken record {i}"
        );
        let p = write_tmp(tmp.path(), &format!("b{i}.json"), &v);
        assert!(
            thme::bench::ResultRecord::load(&p).is_err(),
            "loader accepted broken record {i}"
        );
    }
}

#[test]
fn committed_gate_files_and_broken_copies() {
    let schema = load("docs/schemas/quality-gate-v1.json");
    for file in [
        "configs/gates/example-strict.json",
        "configs/gates/thme-perplexity-v1.json",
    ] {
        let good = load(file);
        assert_eq!(errors(&schema, &good), Vec::<String>::new(), "{file}");
        thme::quality::gate::GateSpec::load(&root().join(file)).unwrap();
    }
    let good = load("configs/gates/example-strict.json");
    let tmp = tempfile::tempdir().unwrap();
    let broken: Vec<Edit> = vec![
        Box::new(|v| v["criteria"] = json!([])),
        Box::new(|v| v["criteria"][0] = json!({"metric": "perplexity", "lower_is_better": true})),
        Box::new(|v| v["criteria"][0]["max_abs"] = json!(-1.0)),
        Box::new(|v| v["name"] = json!("")),
    ];
    for (i, edit) in broken.iter().enumerate() {
        let mut v = good.clone();
        edit(&mut v);
        assert!(
            !errors(&schema, &v).is_empty(),
            "schema accepted broken gate {i}"
        );
        let p = write_tmp(tmp.path(), &format!("g{i}.json"), &v);
        assert!(
            thme::quality::gate::GateSpec::load(&p).is_err(),
            "loader accepted broken gate {i}"
        );
    }
}

#[test]
fn surgery_recipe_example_matches_its_schema() {
    let schema = load("docs/schemas/surgery-recipe-v1.json");
    let good = load("examples/recipes/hybrid_mamba_deltanet_recipe.json");
    assert_eq!(errors(&schema, &good), Vec::<String>::new());
}

#[test]
fn the_pattern_subset_matches_like_a_regex() {
    assert!(pattern_matches("^[0-9a-f]{64}$", &"ab".repeat(32)));
    assert!(!pattern_matches("^[0-9a-f]{64}$", &"ab".repeat(31)));
    assert!(pattern_matches("^[a-z0-9][a-z0-9._-]*$", "toy-arith.v2"));
    assert!(!pattern_matches("^[a-z0-9][a-z0-9._-]*$", "-toy"));
    assert!(pattern_matches(
        "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        "2026-11-13"
    ));
    assert!(!pattern_matches(
        "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        "2026-1-13"
    ));
}
