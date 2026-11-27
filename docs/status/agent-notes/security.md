# Agent notes: v0.2.7 security and trust (branch agent/v02-security)

No detection rates or measurements were taken; every claim below is about what the tests check.
v0.2.7.6 (sandbox) was not touched: shelved by the owner.

## v0.2.7.2 Scanner coverage (THME-3022)

- Done-means: each format added with fixtures; compressed pickles inspected with a pure-Rust
  inflate dependency (flate2, already in the tree).
- Found: deflated pickles, NumPy object arrays (`.npy`, `.npz`), Keras (`.keras`, `.h5`) and ONNX
  were already implemented and tested in `src/scan` (see the BACKLOG progress note). The only
  gap was a systematic truncation check.
- Added: `every_truncation_of_zip_pickle_and_npz_scans_without_panic` in `src/scan/tests.rs`
  (every prefix of a deflated and a stored PyTorch zip and a deflated `.npz` scans without a
  panic; the full files still report Critical).
- Verdict: met. Known limit stays as documented: compressed, chunked or fragmented `.h5`
  attributes can be missed (reported as `hdf5-partial`). The ROADMAP row is still empty; the
  coordinator should mark it.

## v0.2.7.4 Permission and ownership record (THME-3008, WP19)

- Done-means: reverse-engineering features refuse to run without a stored basis and purpose
  statement next to the model hash; wording reviewed by a lawyer.
- Added: `src/permission.rs` (`grant`, `require`, `record_path`, `Basis`). The record is
  `<model>.thme-permission.json` beside the file, holds SHA-256, basis, purpose and time, and
  `require` fails if it is missing, malformed, has an empty purpose, an unknown version, or a
  hash that no longer matches the file. Tests: `refuses_without_a_record`,
  `grant_then_require_round_trips_with_the_known_hash`,
  `empty_purpose_is_refused_and_nothing_is_written`, `changed_model_invalidates_the_record`,
  `hostile_records_are_errors_not_panics`, `folders_and_missing_files_are_refused`.
- Verdict: partly met. Missing: (1) no reverse-engineering feature (architecture inference,
  closed-format analysis) exists in the tree yet, so nothing calls `require`; (2) no CLI command
  to record permission; (3) the lawyer review of the wording. The record is the user's own
  statement and is not verified.

## v0.2.7.7 Guardrail detection (THME-3021)

- Done-means (whole ticket): scan reports safety components; an evaluation reports refusal
  rates with intervals; optional safety layers can be switched on or off. The roadmap row asks
  only for the read-only detection part.
- Added: `src/scan/guardrail.rs`, hooked into `src/scan/mod.rs`. Info findings
  `guardrail-refusal-instruction` (refusal wording in chat templates, GGUF or JSON, and in
  `system_prompt*.txt/md`) and `guardrail-component` (known safety-classifier names in
  `config.json` `_name_or_path`/`architectures`, GGUF `general.name`/`general.basename`, and a
  non-null `safety_checker` in diffusers `model_index.json`). Info never changes the verdict.
  Tests (in `src/scan/tests.rs`): `refusal_instruction_in_gguf_chat_template_is_reported_as_info`,
  `refusal_instruction_in_tokenizer_config_template_and_system_prompt_file`,
  `plain_template_and_prompt_report_no_guardrail`, `safety_classifier_components_are_reported`,
  `disabled_safety_checker_and_gguf_guard_name`, `guardrail_scan_never_panics_on_hostile_json`.
- Verdict: partly met. Detection is keyword and name matching on a short hand-written list, so
  it misses guardrails with other wording or names and may match unrelated text. Not attempted:
  the refusal-rate evaluation with intervals (needs a fixed prompt set and a model run, and the
  owner's call on the prompt set).

## v0.2.7.8 Optional safety layers

- Done-means (roadmap row, ticket THME-3021 last clause): layers switched on or off per
  deployment. Asked here: design plus the smallest tested core.
- Added: `docs/safety-layers.md` (design) and `src/safety.rs` (`SafetyConfig::from_json`,
  `check`; layers `max_input_chars` and `deny_phrases`, off by default, strict config). Tests:
  `everything_is_off_by_default`, `switching_a_layer_off_again_restores_allow`,
  `length_limit_applies_to_input_only_and_counts_characters`, `bad_configs_are_errors`.
- Verdict: partly met. Missing: loading the config in a deployment and calling `check` in the
  serve, chat and MCP paths; classifier-based layers; block logging.

## For the coordinator

- ROADMAP rows 7.2, 7.4, 7.7 and 7.8 and the BACKLOG tickets were not edited (instructed).
- Decide: the prompt set and model for the refusal evaluation; where a deployment config file
  lives; which feature will be the first caller of `permission::require`.
