## What and why

<!-- One or two sentences. Link the backlog ticket: "Closes #123" or "THME-2201". -->

## Roadmap item

<!-- e.g. v0.2.2.4 -->

## Evidence

<!-- Pick the strongest level that applies (see ROADMAP.md, "Evidence levels"). -->

- [ ] **Measured**: new or changed figures come from a report committed under `docs/benchmarks/`, and the dossier and charts were regenerated
- [ ] **Tested**: behaviour is covered by new or updated tests
- [ ] **Estimate / Simulated**: every such figure is labelled as such where it is shown
- [ ] No figures changed

## Checklist

- [ ] `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and `cargo test` pass locally
- [ ] No model weights, binaries or absolute local paths are committed
- [ ] Parsers and public entry points return errors instead of panicking
- [ ] `CHANGELOG.md` has an entry under "Unreleased" for user-visible changes
- [ ] `ROADMAP.md` / `BACKLOG.md` status updated if this completes an item
- [ ] Studio screenshots recaptured (`scripts/capture_studio.ps1`) if the GUI changed visibly
