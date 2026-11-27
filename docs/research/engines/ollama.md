# Survey: Ollama (clone at <engines>\ollama)

Read-only survey, 2026-11-27. Only paths I opened or listed are cited. "Unverified" = not opened.
Effort S/M/L is my estimate, not measured. No source is copied here.

## 1. Licence
- `LICENSE` (21 lines) opens with "MIT License" then "Copyright (c) Ollama" (read, lines 1-3).
- No `NOTICE*` file in the repo root (ls returned nothing).
- Obligation (standard MIT text): keep the copyright line and permission notice in copies or substantial
  portions. If THME ports a file, add an SPDX `AGPL-3.0-only AND MIT` header and put the Ollama text in
  `LICENSES/` as is done for llama.cpp (`LICENSES/llama.cpp-MIT.txt`). Not legal advice.
- Ollama builds on llama.cpp (see section 2), so anything that is really llama.cpp code carries that licence too.
- Model licences and the registry service terms are not the same as this code licence; unverified, not read.

## 2. What it is
Ollama is a Go server and CLI that pulls, creates and runs models behind a local REST API and
OpenAI/Anthropic-compatible endpoints. For GGUF it runs a pinned llama.cpp (`LLAMA_CPP_VERSION` holds
`b12734`; `llama/README.md`, `llama/compat/` patches) as a child server (`llm/llama_server.go`); it also has an
MLX path (`mlx/`, `mlxrunner/`). Its value is packaging (Modelfile, content-addressed registry,
templates, scheduling), not new numerics.

## 3. Components
| Component | What it does | Paths opened |
| :-- | :-- | :-- |
| Modelfile parser | FROM, PARAMETER, TEMPLATE, SYSTEM, ADAPTER, LICENSE, MESSAGE, and more | `parser/parser.go` (778 lines), `docs/modelfile.mdx` |
| Manifest and layers | OCI-style manifest, typed layers (model, template, system, params, adapter, license, messages) | `manifest/layer.go`, `manifest/manifest.go` |
| Pull/push | resumable chunked blob download, digest checked | `server/download.go` (551), `server/upload.go`, `server/images.go` (`PullModel`) |
| Templates | Go `text/template` per model, library of named templates | `template/template.go`, `template/*.gotmpl` + `*.json` (listed) |
| Create and quantize | build model from safetensors/GGUF, quantize | `create/` (listing), `create/quantize.go`, `create/quantpolicy.go`, `server/create.go` |
| Scheduler | load, keep-alive, placement, evict | `server/sched.go` (1853 lines), `envconfig/config.go` |
| Hardware discovery | GPU/CPU memory probing | `discover/` (listing) |
| Runner glue | wraps llama.cpp server, GBNF helper | `llm/server.go` (grepped), `llm/gbnf.go`, `llama/compat/` |
| API layers | native, OpenAI, Anthropic compat | `api/`, `openai/`, `middleware/` (listings) |
| Output parsers | tool-call and thinking parsers per model family | `model/parsers/`, `model/renderers/`, `harmony/`, `thinking/` (listings) |
| MLX engine | native Go/MLX inference, quantized tensors | `mlx/`, `mlxrunner/` (listings only; unverified internals) |

## What THME already has (opened)
- Discovery of Ollama models on disk: `thme discover` (README.md "Plug and play"); a bridge is planned (README, THME-3042).
- Hub download with checksum and licence print: `src/hub/` and `src/cli/download.rs`. HF only; no registry client.
- Chat templates via minijinja: `Cargo.toml` lines 110-113, `src/cli/chat_template.rs`.
- Recipes (declarative per-tensor formats) and `.epiph` single file with provenance: README "Now building".
- No model scheduler or server in `src/` listing.

## 4. What THME lacks, ranked
**1. Reading Ollama's local store (manifest plus blobs) directly. Effort S.**
- Why: users already have models there; lets THME run or convert them with no re-download. Paths: `manifest/layer.go` (media types: `application/vnd.ollama.image.model`, `.template`, `.params`, `.system`, `.adapter`), `manifest/paths.go` (listed), `server/images.go` (line 801-811 switch on layer media type).
- Translate: JSON manifest parse, sha256 blob path, GGUF blob to existing loader; read-only.
- Drags in: none (serde_json, sha2 likely present; unverified). Risk: low; store layout may change between versions.
- Test: `ollama list`, `ollama show --modelfile NAME` output vs THME parse; hash of the blob equals its digest.
**2. Modelfile import (FROM, PARAMETER, TEMPLATE, SYSTEM, MESSAGE). Effort S-M.**
- Why: users keep sampling defaults and templates there; gives THME chat defaults and a recipe-like text format. Paths: `parser/parser.go` (command switch near lines 69-151, validity list line 732), `docs/modelfile.mdx` (sections FROM, PARAMETER, TEMPLATE, SYSTEM, LICENSE, MESSAGE, CAPABILITY, REQUIRES).
- Translate: a small line parser with multi-line quoted strings. Drags in: none. Risk: low.
- Test: `ollama create` then `ollama show --modelfile`; round-trip with THME writer; `parser/parser_test.go` cases as fixtures (read tests only).
**3. Go-template to minijinja translation for Ollama TEMPLATE strings. Effort M.**
- Why: Ollama models without a Jinja chat template carry Go templates (`template/template.go` `Parse`/`Execute`; library `template/*.gotmpl`). Only needed for models whose GGUF lacks `tokenizer.chat_template`.
- Translate: pick the matching named template by similarity, or render a small Go-template subset. Drags in: a Go-template subset interpreter (no crate assumed). Risk: medium (full text/template semantics are large). Test: `ollama run` with `--verbose` or `/api/generate` `raw` prompt dump vs THME-rendered prompt; named fixtures `template/*.json` hold expected values (opened as listing only).
**4. Registry pull protocol (v2 manifests and blobs, ranged, resumable). Effort M.**
- Why: gets models without Hugging Face; shows the manifest design (typed layers, digest-addressed blobs, dedupe). Paths: `server/download.go` (Range header near line 378, retry loop line ~299, URL join `v2/<ns/model>/blobs/<digest>` line 540), `server/images.go` (manifest URL line 1114, `PullModel` line 1193), `server/auth.go` (token auth; unverified detail).
- Translate: manifest GET, parallel ranged blob GET, sha256 verify, token auth. Drags in: HTTP client already in `src/hub/http.rs`. Risk: medium (auth, protocol drift, registry terms unverified; fetching their weights is a licence matter per model).
- Test: pull the same small model with `ollama pull`, compare blob digests with THME's.
**5. Quantization policy by tensor name and shape. Effort S (ideas only).**
- Why: a small, readable rule table to benchmark THME's measured search against. Paths: `create/quantpolicy.go` (default policy delegates to `GetTensorQuantization`, layer-index regex), `create/quantize.go`. Note the formats there are MLX-style block formats packed to safetensors (`blockfp8.go`, `mlx/quant/` listed), not GGUF K-quants; for GGUF Ollama calls llama.cpp's quantizer (inferred from `server/create.go` quantize fields; unverified path).
- Risk: low. Test: compare the chosen per-tensor types against THME's own plan output for the same model.
**6. Multi-model scheduling in memory. Effort L.**
- Why: Ollama keeps several models loaded with a keep-alive timer, a queue, and predicted-VRAM placement (`server/sched.go` `processPending` 274, `processCompleted` 416, expire timer lines 435-458, placement functions near 1005-1160; env `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_QUEUE`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION` in `envconfig/config.go`).
- Translate: only the idea (LRU-by-idle plus memory-fit check). THME's `thme pathfinder` already predicts weights/KV/scratch bytes (README), which is the same input. Only worth building if THME gets a server.
- Risk: medium. Test: scripted requests to two models with `ollama ps` as the oracle for what is resident.
**7. Structured output through grammar. Effort M, depends on llama.cpp survey item 6.** `llm/gbnf.go` composes a thinking-aware grammar on top of a format grammar; useful only after THME has GBNF.

## 5. What NOT to port
- The runner and server layer: Go, REST, cloud proxies (`server/cloud_proxy.go`, `server/codex_proxy.go`), web search middleware, auth. THME has no server and should not start there.
- OpenAI/Anthropic compat layers (`openai/`, `middleware/`): large API surface, no quantization value.
- `llama/compat/` patches and the llama.cpp build glue: it is llama.cpp; port from llama.cpp directly.
- MLX engine (`mlx/`, `mlxrunner/`) and its per-model Go files in `create/` (dozens of architectures): Apple-oriented, Go/C bindings, high maintenance.
- Output parsers and renderers per model family (`model/parsers`, `model/renderers`, `harmony`): tool-calling product features, not engine measurement.
- GPU discovery (`discover/`): THME has `thme doctor` and a wgpu adapter probe.
- Registry weights themselves: pulling third-party models raises per-model licence checks; THME's download already prints the licence first.
