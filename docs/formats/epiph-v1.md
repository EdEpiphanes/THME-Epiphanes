# The Epiphanes model format (`.epiph`), version 1.3

**Status:** draft, open specification. The reference reader and writer are `src/epiph/` in this
repository (AGPL-3.0-only). Anyone may implement this specification.
**Name:** "Epiphanes" and the format's name are trademarks of Epiphanes; registration pending
legal review. The `.epiph` extension is a label, not a trademark claim.

## 1. Purpose

One file holds a whole model and everything known about it: its weights in any THME format,
its architecture and configuration, the recipe that produced it, where it came from, the quality
gate it passed, and the calibration statistics needed to change it again. A reader never needs a
sibling file, and every claim in the file carries its evidence level.

Design rules:

- **Self-describing.** No sidecar `config.json`; a reader ignores files next to it.
- **Zero-copy.** Every payload starts on a 64-byte boundary, so it can be memory-mapped.
- **Verifiable.** Every payload has a SHA-256 digest in the manifest.
- **Extensible without breaking.** Unknown manifest fields are ignored; unknown header flags
  and newer major versions are refused.
- **No hidden locations.** The manifest never stores an absolute path or a user name.

## 2. Layout

All integers are little-endian.

| Offset | Size | Field | Value |
| ---: | ---: | :--- | :--- |
| 0 | 8 | magic | `89 45 50 49 50 48 0D 0A` (`\x89EPIPH\r\n`) |
| 8 | 2 | major version | `1` |
| 10 | 2 | minor version | `3` |
| 12 | 4 | flags | `0` (all bits reserved; a reader refuses a file with any bit set) |
| 16 | 8 | manifest length `M` | bytes of UTF-8 JSON |
| 24 | 8 | data offset `D` | multiple of 64, at least `32 + M` |
| 32 | `M` | manifest | UTF-8 JSON object (section 3) |
| `32 + M` | `D - 32 - M` | padding | zero bytes |
| `D` | rest | data | payloads, each at `D + offset`, `offset` a multiple of 64 |

The first byte is non-ASCII and the magic contains CR LF, so text-mode transfers and
truncation to 7 bits are detected, as in PNG.

A reader refuses: a wrong magic; a major version above the one it implements; any flag bit set;
a manifest that is not valid UTF-8 JSON; a payload that overlaps another or runs past the end of
the file; a data offset that is not a multiple of 64; a tensor `encoding` it does not know (1.3).

## 3. Manifest

```json
{
  "format": "epiph",
  "version": "1.3",
  "model": { "...": "the model description (section 3.1)" },
  "tensors": [
    { "name": "blk.0.attn_q.weight", "offset": 0, "length": 1024, "sha256": "...", "encoding": "deflate", "decoded_length": 4096, "decoded_sha256": "..." },
    { "name": "blk.0.attn_k.weight", "offset": 1024, "length": 4096, "sha256": "..." }
  ],
  "attachments": [ { "kind": "calibration", "encoding": "json", "offset": 4096, "length": 900, "sha256": "..." } ],
  "recipe": null,
  "quality": [],
  "provenance": { "tool": "thme", "tool_version": "0.2.0", "created_unix": 0, "sources": [] },
  "lineage": { "parents": [] },
  "annotations": {}
}
```

### 3.1 `model`

The THME model description: name, architecture family, dimensions, normalization epsilon, RoPE
base, and one entry per tensor with its name, role, layer index, shape, stored data type
(`original_dtype`), selected format, sensitivity score and per-tensor metadata. In version 1.0
this is the serialized THME intermediate representation; fields a reader does not know are
ignored.

### 3.2 `tensors`

One entry per tensor in `model`, in the same order, giving where its payload lives in the data
section and its SHA-256 digest (lowercase hex). The payload's encoding is the tensor's stored
data type: `F32`, `F16`, `BF16`, or a GGML block type name (`Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`,
`Q6_K`, `TQ1_0`), laid out exactly as in GGUF. A packed recipe candidate stores each tensor in its
selected format; the tensor's metadata then records its type before packing under
`epiph.source_dtype`.

**Payload encoding (1.3).** A tensor entry may carry `encoding`, one of:

- `identity` (the default; the field is then omitted): the payload is stored as-is.
- `deflate`: the payload is stored as a raw DEFLATE stream (RFC 1951; no zlib or gzip wrapper).
  Such an entry also carries `decoded_length`, the length of the decoded payload, and
  `decoded_sha256`, its SHA-256 digest.

For every entry, `length` and `sha256` describe the stored bytes. For a `deflate` entry,
`decoded_length` must equal the tensor's `byte_size` (section 3.1), and the decoded bytes must
match `decoded_sha256`. A writer stores a payload compressed only when the stream is smaller
than the payload, so incompressible data stays `identity`. Stored payloads keep the 64-byte
alignment of section 2.

A reader refuses a `deflate` entry whose `decoded_length` exceeds 1032 times its `length` (the
largest expansion DEFLATE can give) or exceeds 16 GiB. A stream that is truncated, that decodes
to other than `decoded_length` bytes, or that is followed by extra bytes is an error. A reader
decodes a compressed payload only when it is needed, into memory bounded by `decoded_length`;
`identity` payloads are still read in place.

### 3.3 `attachments`

Additional payloads with a `kind` and an `encoding`. Version 1.0 defines:

| kind | encoding | contents |
| :--- | :--- | :--- |
| `calibration` | `json` | per-input-channel calibration statistics, as written by `thme calibrate` |
| `gguf-metadata` | `json` | every GGUF metadata key of the source, each value tagged with its exact type (`{"Uint32": 7}`), including the tokenizer; restored on read, so GGUF to `.epiph` to GGUF is lossless (1.1) |
| `hf-file` | `raw` | a Hugging Face file (`config.json`, `tokenizer.json`, ...) stored under `name`, which must be a plain file name; `thme epiph extract` writes these back (1.1) |

Attachments may carry a `name` (1.1). Reserved for later minor versions: `report`. A reader skips kinds
it does not know.

### 3.4 `recipe`, `quality`, `provenance`, `lineage`, `annotations`

- `recipe`: the recipe JSON that produced this model, or `null`.
- `quality`: zero or more quality-gate results. Each records the gate version and criteria, the
  measured values, the verdict, and an `evidence` label (`measured`, `tested`, `estimate`,
  `simulated`, or `imported` for a report copied in from elsewhere and not re-measured). A reader must not present a result whose label is not `measured` as a
  measurement.
- `provenance`: the writing tool and version, creation time (Unix seconds, UTC), and one entry
  per source file with its file name (no directory) and SHA-256 digest.
- `lineage.parents`: SHA-256 digests of the `.epiph` files this one was derived from, so a chain
  of edits can be followed.
- `annotations`: free string key-value pairs.

### 3.5 `placement`

Optional. Present only when the file was packed with a memory budget; otherwise the key is absent.
An object with `budget` (`vram_bytes`, `host_bytes`), the byte totals `gpu_bytes`, `cpu_bytes`
and `stream_bytes`, and `tensors`: one entry per manifest tensor, in manifest order, with `name`,
`bytes`, `tier` (`gpu`, `cpu` or `stream`) and `reason`. THME's exact allocator plans it on the
stored payload sizes. A reader that does not know this section ignores it. It says where a runtime
should put each tensor; it says nothing about measured speed.

## 4. Verifying a file

A file is valid when the header and manifest pass the checks in section 2, every payload's
SHA-256 matches its manifest entry, every `deflate` payload decodes to its `decoded_length` with
a matching `decoded_sha256` (section 3.2), and each tensor payload's length matches its shape and
stored data type. `thme epiph verify FILE.epiph` performs these checks and names the tensor when
one fails.

## 5. Converting

Any model THME can import (SafeTensors directory, GGUF file) can be packed with
`thme epiph pack`. A `.epiph` file is imported like any other model, so `thme export` writes it
to GGUF or SafeTensors, and every other command reads it directly.

## 6. Signature

A file may end with a Minisign signature over the **manifest bytes**. Because the manifest holds
every payload's SHA-256, the signature binds the whole file. Layout, after the last payload
rounded up to 64 bytes:

| Field | Size |
| :--- | :--- |
| Minisign signature text (prehashed Ed25519, as written by Minisign) | `S` |
| `S` as an unsigned 64-bit little-endian integer | 8 |
| `EPIPHSIG` | 8 |

The trusted comment is `epiph-manifest-sha256:<hex>`. Readers that do not check signatures
ignore the trailer. Signing again replaces it. Official Epiphanes files are signed with the key in
`minisign.pub`; `thme epiph verify` checks against that key unless another is given, and
`--require-signature` makes an unsigned or unverifiable file an error. Converters that do not
implement this section still read the weights; they lose only the signature.

## 7. Planned extensions (not in 1.0)

Each lands as a minor version with tests before it is used:

- Optional larger payload alignment (4096 bytes) for page-aligned transfers.

## 8. Versioning

A minor version adds optional manifest fields or attachment kinds; older readers still read the
file. A major version may change the layout; readers refuse majors they do not implement.

| Version | Adds |
| :--- | :--- |
| 1.0 | Layout, manifest, tensors, attachments, signature trailer |
| 1.1 | `name` on attachments; `gguf-metadata` and `hf-file` kinds |
| 1.2 | `placement` section |
| 1.3 | Per-tensor `encoding` (`identity` or `deflate`), with `decoded_length` and `decoded_sha256` for `deflate` |

Changes: 1.1 added `name` on attachments and the `gguf-metadata` and `hf-file` kinds. 1.2 adds
`placement`. 1.3 adds the per-tensor `encoding` for lossless compression of tensor payloads.
Shortcut: a 1.2 or older reader ignores `encoding` and is not tested to refuse a compressed file;
it must not be given one. Only 1.3 readers are expected to decode `deflate` payloads.

## 9. Portability

Verified by `tests/test_epiph_portability.rs`:

- The layout in section 2 is little-endian by specification. The golden hashes in the test are asserted on the
  three CI hosts: Linux x86-64, Windows x86-64 and macOS arm64.
- For a fixed model (built from fixed integers, with no clock, random value or host path), the
  writer produces the same bytes on each of those hosts. The SHA-256 of the whole file with
  identity encoding is pinned, and so is the SHA-256 of the decoded payloads of a `deflate`
  file.
- Reading the packed file back reproduces every tensor's name, shape and bytes exactly.

Not verified:

- Big-endian hosts. The layout is little-endian by specification, but no big-endian run exists.
- Chips other than x86-64 and arm64, and 32-bit targets.
- The DEFLATE stream bytes. Section 3.2 fixes the decoded payload, not the compressed stream,
  so the test pins only the decoded bytes.

The golden hashes change when the writer or its tool version changes. They are updated
deliberately, in the same change as the writer.
