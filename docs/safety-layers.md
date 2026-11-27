# Safety layers (v0.2.7.8)

Status: design plus a tested core (`src/safety.rs`). No serving path calls it yet.

## What a safety layer is

A safety layer is a check THME adds around a model in a deployment. It looks at text going in or
coming out and can block it. It is separate from the model: THME never edits weights, never
removes or weakens a model's own safety behaviour (out of scope per THME-3021), and a layer that
is switched off leaves the model exactly as it was.

## Rules

1. Layers are off by default. A deployment switches each one on in its config.
2. The config is strict: an unknown layer or field is an error, so a typo cannot leave a layer
   silently off. An enabled layer must be usable (a limit above zero, at least one phrase).
3. Layers run in a fixed order and the first block wins. A block names the layer and gives a
   reason that is safe to show to the user.
4. Layers are simple and inspectable. They are not a claim of safety; a clean pass means no layer
   blocked, not that the text is safe.

## Layers in the core

| Layer | Direction | Blocks when |
| :--- | :--- | :--- |
| `max_input_chars` | input | the text has more than `limit` characters |
| `deny_phrases` | input and output | the text contains one of `phrases` (case-insensitive) |

## Config

```json
{
  "max_input_chars": { "enabled": true, "limit": 8000 },
  "deny_phrases": { "enabled": false, "phrases": ["example phrase"] }
}
```

## Not built yet

- Loading the config from a deployment file or CLI flag and calling `SafetyConfig::check` in
  `thme serve`, chat and the MCP server.
- Classifier-based layers (they need a model and an evaluation of their error rates first; THME
  states no detection rates until they are measured).
- Logging of blocks.
