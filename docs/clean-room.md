# Clean-room process

How THME learns from other projects and papers and builds its own, original implementation. It
is a working rule for people and agents, written down so that every claim of "original work" can
be checked. The shape of the loop (study, plan, build, check, refine, with a strict rule about
what the builder may see) follows the published workflow of the open-source project
[Craft Anything](https://github.com/storytold/craft-anything); the adaptation to inference engines
is ours. This page is a process document, not legal advice.

## The loop

1. **Study** from outside. Read public explanations (documentation, format specifications, design
   discussions, papers, vendor documentation) and measure the other system as a black box with its
   own public tools (benchmark and profiling switches). Do not read or copy its source code or
   shader files.
2. **Design notes.** Write what was learned as ideas, trade-offs and measured behaviour, never as
   code, and add our own hypotheses with the arithmetic and a "measured how" for each.
3. **Build** from the design notes and from THME's own profiles. The builder works in original
   Rust and WGSL.
4. **Check** against the reference: the same inputs, the same quality gate, throughput and memory
   on the same machine, recorded with the device, driver, build and load.
5. **Refine**, one change at a time with a paired before and after, until the checks pass.

## Rules

- Allowed sources: public documentation, specifications, papers and their prose, vendor
  documentation, public benchmark outputs, black-box measurements we make ourselves.
- Not allowed for the builder: the other project's source code, shader files, decompiled output or
  anything derived directly from them. Ideas and facts (a published file format, a mathematical
  method) may be used; expression (code, comments, structure copied from a file) may not.
- Permissive licences do not change this rule. If code from another project is ever included, it
  goes through the reference audit first: licence read from the file, notices kept, an entry in the
  third-party register and the provenance ledger (THME-3071), and the owner's approval.
- Keep a record for every learning step: which public sources were read, by whom or which agent,
  on what date, and which notes came out. A person or agent who has read the other project's
  source code does not write the replacement for that part.
- Anything unmeasured is labelled hypothesis, reported (by a third party) or simulated.

## What this looks like for llama.cpp

Design notes live in the private research folder; THME measures llama.cpp through its own benchmark
tool (see the reference report in `docs/benchmarks/smollm2-1.7b/data/`); the GPU work (THME-3086 to
3089) is built from those notes and from THME's own kernel profiles; interop with GGUF follows the
public format specification and is tested against the real consumer (THME-3091).
