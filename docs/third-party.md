# Third-party code register

This register lists every outside source used by this repository, so the origin, licence,
files involved and the notice that must stay with them can be checked in one place.
Licences are read from each source's own licence text — see `LICENSING.md` for the
project-wide model and `LICENSES/anatomy-MIT.txt` for the anatomy figure kit's full MIT
text. Code under GPL, non-commercial or "do not copy" terms is never copied: the idea is
rewritten from the paper. Tracked files whose header names a non-AGPL licence must have a
row here (enforced by `scripts/third_party.py --check`).

`THIRD_PARTY_LICENSES.html` in release archives is generated from `about.hbs` and
`cargo about generate`, and covers the crates below. The crate licences below were checked
against `cargo metadata` on 19 November 2026.

| Name | URL | Licence | What was reused | Notice kept | Files |
| :--- | :--- | :--- | :--- | :--- | :--- |
| minisign | https://crates.io/crates/minisign | MIT | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| flate2 | https://crates.io/crates/flate2 | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| resvg | https://crates.io/crates/resvg | Apache-2.0 OR MIT | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| gif | https://crates.io/crates/gif | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| wgpu | https://crates.io/crates/wgpu | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| eframe (egui) | https://crates.io/crates/egui | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| safetensors | https://crates.io/crates/safetensors | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| memmap2 | https://crates.io/crates/memmap2 | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| tokenizers | https://crates.io/crates/tokenizers | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| minijinja | https://crates.io/crates/minijinja | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| minijinja-contrib | https://crates.io/crates/minijinja-contrib | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| Chakra Petch font | https://github.com/m4rc1e/Chakra-Petch.git | SIL Open Font License 1.1 | Bundled font for the GUI; no source copied | `assets/fonts/OFL.txt` | `assets/fonts/ChakraPetch-Bold.ttf`, `assets/fonts/ChakraPetch-Medium.ttf` |
| anatomy figure kit | https://github.com/wheresryan22/anatomy | MIT | `src/engine/iso.rs` ported from the kit's core (TypeScript) to Rust; `assets/iso.css` copied | File headers plus `LICENSES/anatomy-MIT.txt` (see `LICENSING.md`) | `src/engine/iso.rs`, `assets/iso.css` |
