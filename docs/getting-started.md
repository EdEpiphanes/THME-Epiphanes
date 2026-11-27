# Getting started

THME is one command-line program, `thme`, and one desktop window, the Studio. This page covers
installing them and three short guides, one for each kind of user. Every command shown here is in
`thme --help` or the help of its subcommand. Anything that does not exist yet is marked **planned**.

## Install

There is no one-command installer and no package-manager package yet. The steps are:

1. Open the **Releases** page of the repository on GitHub and download the archive for your
   machine. The release pipeline ([release.yml](../.github/workflows/release.yml)) builds four:

   | Machine | Archive |
   | :--- | :--- |
   | Linux, x86_64 | `thme-<tag>-x86_64-unknown-linux-gnu.tar.gz` |
   | Windows, x64 | `thme-<tag>-x86_64-pc-windows-msvc.tar.gz` |
   | macOS, Apple silicon | `thme-<tag>-aarch64-apple-darwin.tar.gz` |
   | macOS, Intel | `thme-<tag>-x86_64-apple-darwin.tar.gz` |

   Each archive comes with a `.sha256` file of the same name. Every release also has a combined
   `SHA256SUMS.txt`. Archives are `.tar.gz` on every platform. Windows 10 and later ship `tar`.

2. Check the SHA-256 checksum. On Linux, in the folder with the files:

   ```bash
   sha256sum -c thme-<tag>-x86_64-unknown-linux-gnu.tar.gz.sha256
   ```

   On macOS use `shasum -a 256 -c` with the same file. On Windows, run
   `certutil -hashfile thme-<tag>-x86_64-pc-windows-msvc.tar.gz SHA256` and compare the hash
   with the one in the `.sha256` file.

3. Check the Minisign signature, if the release has a `.minisig` file for your archive. Get
   `minisign.pub` from the repository at the same tag, then run:

   ```bash
   minisign -Vm thme-<tag>-<target>.tar.gz -p minisign.pub
   ```

   Minisign is a separate tool and is not part of THME. The checksum only shows the file is intact.
   The signature shows it came from the key in `minisign.pub`. Releases are signed only when the
   signing key is configured for the run; if a release has no `.minisig` files, rely on the checksum
   and say so in your notes.

4. Unpack the archive. It holds a folder with `thme` (`thme.exe` on Windows), `thme-gui` (the
   Studio), the licence and notices, and the changelog:

   ```bash
   tar -xzf thme-<tag>-<target>.tar.gz
   ```

5. Put both programs in one folder and put that folder on your PATH. Keep `thme` and `thme-gui` in
   the same folder: `thme gui` starts the Studio by looking for `thme-gui` next to `thme`.

6. Check the install:

   ```bash
   thme --version
   thme doctor
   ```

**Build from source instead.** Install Rust with [rustup](https://rustup.rs/). The repository pins
its toolchain in `rust-toolchain.toml`, and rustup fetches it on first use. Then, in the repository
folder:

```bash
cargo build --release
```

The program is written to `target/release/thme` (`thme.exe` on Windows), and `thme-gui` next to it.

Nothing here has been tried on a clean machine yet.

## Data analyst: find models, look inside them, draw them, chat

1. **Find the models already on your machine.** THME reads the Hugging Face cache, Ollama, LM
   Studio and llama.cpp folders, and any folder you name. Nothing is copied or moved.

   ```bash
   thme discover
   thme discover --path path/to/my/models
   ```

2. **Check what this machine can do.** `thme doctor` lists the system, memory, disk, GPU adapter and
   the models it found, and says what to try next.

   ```bash
   thme doctor
   ```

3. **Inspect a model.** Give it a model folder or a SafeTensors, GGUF or `.epiph` file. It reads
   the header; it does not copy the weights. `profile` adds the layer sensitivity table and how the
   KV cache grows with the context length.

   ```bash
   thme inspect path/to/model
   thme profile path/to/model
   ```

4. **Draw a model.** `thme figure` writes one dark and one light SVG, with one tray per layer and
   one block per tensor. Add `--gif` for animated versions.

   ```bash
   thme figure path/to/model --out figures
   ```

5. **Chat with a model in the Studio.** Start the Studio with `thme gui`, load a model, and open the
   Chat tab. It uses THME's own CPU forward pass and the model's chat template. It is slow by design,
   and it has been tested on the synthetic mock model only, not yet on a real model.

   ```bash
   thme gui
   ```

## AI engineer: scan, convert, export, serve to editors and agents

1. **Scan a model before you open it.** The scan looks for code that would run on load, hidden
   payloads and malformed headers. It executes nothing. Use `--report` to keep the findings as JSON.

   ```bash
   thme scan path/to/model --report scan.json
   ```

   The command fails when the worst finding is `high` or worse. Change that with `--fail-on`.
   A clean scan is not proof that a model is safe.

2. **Export to GGUF or SafeTensors.** `--quant` picks the weight type (`fp16`, `q8_0`,
   `bitnet-b1.58`, or `per-tensor`). Add `--checksum` to print a SHA-256 of the result.

   ```bash
   thme export path/to/model -o model.gguf --format gguf --quant q8_0 --checksum
   ```

   `thme convert --to llm|lcm|lam|moe|vlm|slm|mlm|sam` is a different job: it reshapes the model's
   topology between paradigms and reports the size change. Without `--model` it uses a built-in
   synthetic baseline.

3. **Serve THME to editors and agents.** `thme mcp` runs a Model Context Protocol server over
   stdin and stdout. `-m MODEL` binds a default model to the session. The editor setup is in the
   [VS Code and MCP section](../editors/vscode/README.md#option-b-universal-mcp-server-for-any-ide),
   which uses the command `thme` with the argument `mcp`.

   ```bash
   thme mcp --model path/to/model
   ```

4. **Download a model from the Hugging Face hub.** **Planned, not built yet** (ROADMAP v0.3.0.5).
   Until it lands, download the model with your own tools and point THME at the folder.

## Data scientist: notebooks, evaluation, recipes

1. **Use THME from a notebook.** Every command below that prints data has a `--json` flag, and
   a notebook reads its output with `json.loads`. See [notebooks.md](notebooks.md) for the list
   of commands and their keys, and [quickstart.ipynb](../examples/notebooks/quickstart.ipynb) for a
   first notebook.

2. **Measure perplexity with the reference forward pass.** `thme evaluate` takes a surgery recipe
   and a token file and reports the perplexity change. The token file comes from
   `scripts/prepare_eval_tokens.py`. `thme search` picks a mixed-precision recipe from measured
   costs under a size budget. `thme forward` prints the logits for a list of token ids.

   ```bash
   thme evaluate path/to/model --recipe recipe.json --tokens tokens.bin
   thme search path/to/model --tokens tokens.bin --budget 0.5
   thme forward path/to/model --tokens 1,2,3 --json
   ```

   THME has no published real-model measurements yet. Results from these commands are measurement
   pending until a committed report under `docs/benchmarks/` backs them.

3. **Check a result against a gate.** `thme validate` compares a baseline and a candidate, either as
   benchmark scores or as perplexities:

   ```bash
   thme validate --baseline 84.5 --candidate 83.9
   thme validate --baseline-ppl 10.0 --candidate-ppl 10.2 --json
   ```

4. **Write and apply a recipe.** A recipe is a JSON file; its format is in
   [docs/schemas/surgery-recipe-v1.json](schemas/surgery-recipe-v1.json). `thme tweak` applies one to a
   model. `thme export --recipe` exports a model with the formats the recipe names, and
   `thme search --emit-recipe` writes the searched assignment as a recipe.

   ```bash
   thme tweak --model path/to/model --recipe recipe.json
   thme export path/to/model -o model.gguf --recipe recipe.json
   ```

## Where to go next

- [notebooks.md](notebooks.md): the JSON output of each command.
- [manual.md](manual.md): the full command reference.
- [README.md](../README.md): what THME measures, and what is still measurement pending.
