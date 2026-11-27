# Agent notes: v0.2.6 Epiphanes model format (sub-milestones 5, 6, 8, 9)

Branch `agent/v02-format`. Scope: `src/epiph/` and `tests/test_epiph_portability.rs` only.
ROADMAP.md, BACKLOG.md, CHANGELOG.md and README.md were not edited.

Gate at commit `6e5f8f2`: `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings`
clean; `cargo test -- --skip medgemma` 879 passed, 0 failed across all test binaries;
`py scripts/spdx.py --check` passes. Real-model tests were not run (they need
`THME_REAL_MODEL_TESTS=1`, which is not set). Build: `CARGO_TARGET_DIR` set to the worktree's
`target-nodbg`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_BUILD_JOBS=2`.

Test-first note: the new tests were written before the code they exercise. The first build
failed on a missing import in the new code, and the first green run came after the
implementation. No red run of the new tests was captured, so "failing first" is not
demonstrated by a log.

## v0.2.6.5 Streaming writer and memory-mapped reader

**Added**
- `src/epiph/mod.rs`: `write` now encodes every tensor twice. Pass one records the stored
  length and digests and drops the bytes. Pass two encodes again and streams the bytes to the
  file through a `BufWriter`. If pass two produces different bytes, the write fails and names
  the tensor. Helpers: `encode_tensor`, `pad`, `map_file`, `slice_of`.
- `verify` and `read_attachments` memory-map the file read-only and check payloads in place,
  with no per-payload copy into a `Vec` for identity payloads.
- Tests: `payload_slices_stay_inside_the_file_image` (`src/epiph/tests.rs`).
- Regression oracle, unchanged: `tests/test_epiph_portability.rs`
  (`packed_bytes_match_the_golden_hashes_on_every_host`). The golden file hash and decoded
  payload hash are the same as before this change, so the byte layout did not change.

**Done-means (roadmap row):** "Streaming writer and memory-mapped reader for models larger than
memory."

**Verdict: partly met.**
- Met by construction: the epiph writer holds one tensor payload in memory at a time. Before
  this change it held all of them. Attachments (calibration, GGUF metadata, tokenizer files)
  are still held in memory, which is expected to be small.
- Missing: nothing was run on a model larger than memory, and peak memory was not measured.
  No claim about it is made here.
- Missing: the inference read path (`src/inference/weights.rs`, `read_tensor_bytes` and
  `open_payload`) still uses seek and read, not a map. Only `epiph::verify` and attachment
  reads map the file. Changing the shared path touches every source format and has a Windows
  locking consequence (a mapped file cannot be overwritten by the same process). Left to the
  coordinator.
- Cost: packing a quantized model encodes every tensor twice. Not measured.
- Shortcut recorded in code: no file lock, so a file rewritten while mapped can show torn
  bytes; the SHA-256 checks reject them.
- The exporter's streaming chunk I/O (v0.1.8.9) is for GGUF and SafeTensors output. It is a
  different writer and was not changed.

## v0.2.6.6 Lossless payload compression behind a per-payload encoding field (THME-3015)

**Already in the repository before this work (format 1.3):** `encoding` on each tensor entry
(`identity` or `deflate`), `decoded_length` and `decoded_sha256` for `deflate`, bounded
inflation (1032 times the stored length, 16 GiB cap, exact length and no trailing bytes),
and `thme epiph pack --compress deflate`. Existing tests include the decompression-bomb case,
truncated and padded streams, and digest mismatches.

**Added here:** nothing for this sub-milestone's code. `describe` (below) reports the
compressed tensor count.

**Done-means (THME-3015):** "Dependency proposed and approved (licence, binary size); per-payload
`encoding` field; compression ratio and decode speed reported as measured on the synthetic
checkpoint only."

**Verdict: partly met.**
- Met: per-payload `encoding` field, lossless round trip, bounds on decompression.
- Missing: no committed measurement of compression ratio or decode speed. I did not run one,
  so no figure is given here. Any figure needs a committed report under `docs/benchmarks/`
  (AGENTS.md section 2, rule 2).
- Missing: no owner approval record for the dependency was found in the repository, so
  "approved" is unverified. `flate2` 1.1.11 is already a dependency (scan of zip pickle
  entries). Its manifest in the local registry states `MIT OR Apache-2.0`. Binary size impact
  was not measured.
- Missing: the spec says a 1.2 or older reader ignores `encoding` and is not tested to refuse
  a compressed file. No older reader exists in this code, so this is untested.

## v0.2.6.8 Studio support (data function only; the view was not attempted)

**Added:** `epiph::describe(path)` in `src/epiph/mod.rs`, returning a `Description`: format
version, tool and tool version, creation time, source file names and digests, lineage parents,
quality entries exactly as written, tensor and compressed-tensor counts, attachment kinds,
whether a placement plan is present, and the signature status checked against the Epiphanes
key. It reads the header and manifest only.
- Test: `describe_summarises_a_file_without_reading_its_payloads` (`src/epiph/tests.rs`). It
  also flips the last byte of a payload: `describe` still succeeds and `verify` fails.

**Verdict:** the data function is met for the fields listed. The Studio view itself was not
attempted: it needs a person to look at the screen. Quality entries are returned unchanged
with their `evidence` label; the view must show only `measured` as a measurement (spec
section 3.4).

## v0.2.6.9 Compatibility report per model and runtime (WP18)

**Verdict: not attempted.** The done-means needs three things this work cannot provide:
1. A configured external runtime (llama.cpp and Ollama). THME-3068 records that the runtime
   check is not configured in CI and was not verified. Nothing was installed or run.
2. A real model. Downloads are not allowed for this task, and the only models are synthetic.
3. A measured gate run for "passes the gate". "Agrees" needs a reference output from the
   runtime to compare with.

The in-process `verify` and `describe` checks are not a runtime load and are not a
compatibility result. No code was written for this item.

## Decisions for the coordinator

1. Whether the inference read path should map files too (v0.2.6.5 partly met until then). It
   changes every file-backed format and has the Windows same-path write consequence above.
2. Whether a committed compression report is wanted (v0.2.6.6). It needs a run on the
   synthetic checkpoint and a file under `docs/benchmarks/`.
3. Whether the `flate2` dependency needs the owner approval that THME-3015 asks for, and
   whether it is recorded in `docs/third-party.md`. This was not checked.
4. Scope of v0.2.6.9: which runtimes and which model, and whether THME-3068 comes first.
5. `target-nodbg/` (the worktree's build directory) is not covered by `.gitignore`, because the
   existing `/target` pattern matches only `target`. I did not change `.gitignore`. Add
   `target-nodbg/` if the convention is wanted.
