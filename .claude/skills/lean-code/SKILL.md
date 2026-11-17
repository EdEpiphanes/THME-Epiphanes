---
name: lean-code
description: Token-efficient coding workflow for sonnet. Use for any implement, fix, refactor, debug, or test task in a codebase, even small edits. Finds the smallest change, reads only what is needed, verifies with the narrowest test first, and hands review to existing skills (simplify, code-review, engineering:*). It prevents over-reading, over-writing, and rework.
---

# Lean code

Smallest correct diff, verified once, reported briefly. Pair with `lean-orchestrate` for routing.

1. **Define done** in one line: the test or command that must pass.
2. **Locate**: `Grep` or `Glob`, then `Read` only the needed range. Project instructions (AGENTS.md / CLAUDE.md) are already in context, so do not re-read them.
3. **Plan** in at most 5 lines, inline. Consult the opus advisor only when the change touches more than 3 files, a public API, a data migration, or concurrency or security code.
4. **Edit**: use `Edit` on existing files, `Write` only for new ones. Keep the diff minimal: no drive-by refactors and no new abstraction before its second use. Match local style and the project's rules.
5. **Verify narrowest first**: one test or file, then the package, then the full suite once at the end. Cap output (`2>&1 | tail -30`). Run the project's required gates once before finishing (this repo: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`).
6. **Debug**: reproduce, form one hypothesis, run one probe. After 2 failed hypotheses, send the evidence to the opus advisor instead of guessing a third time.
7. **Review**: run `simplify` on the diff for reuse and cleanup. Run `code-review` only for risky changes. Delegate reading-heavy audits to haiku with a size-capped brief.
8. **Finish** in at most 5 lines: what changed, the real verification output (state failures plainly), follow-ups. Commit only when asked.

Defects to avoid, since each one costs rework: reading whole files for one function, rewriting a file to change a line, running the full suite after every edit, and fixing symptoms without reproducing first.
