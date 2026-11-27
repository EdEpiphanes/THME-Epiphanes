# Epiphanes Math paper

The research paper behind THME's quality gate and allocator: *Epiphanes Math: A Formal Specification and
Measured Evaluation of the THME Compression and Inference Engine* (Ed Epiphanes).

| File | What it is |
| :--- | :--- |
| `THME_Epiphanes_Paper_v5.1.md` | **Current.** Version 5 plus the changes listed in its Appendix A ("Version 5.1 Changes") |
| `THME_Epiphanes_Paper_v5.md` | Version 5 as the author supplied it, with no edits (git may normalize its line endings) |
| `THME_Epiphanes_Paper_v5.pdf` | The author's PDF of version 5 (not regenerated for 5.1) |

## What version 5.1 changes

- **Section 3.8 (new).** The acceptance bands of US EPA Performance Specification 16 (relative accuracy test and
  audit, bias, variance and correlation tests, three operating levels) transposed to the gate's terms, with the
  committed results re-judged under the literal bands.
- **Section 7.9, item 4.** The loop breaker stops generation, as the code does; the text no longer claims a
  temperature increase and a DoLa reset.
- **Section 8.7.** The release binary size measured on 27 November 2026.
- **Section 11.6 (new).** The measurements committed since version 5, each pointing at its report under
  `docs/benchmarks/`.
- Entries in Sections 12.3 and 13.1, a reference, and a change table in Appendix A.

## What is missing from this folder

- **Figures.** Version 5 links `figures/*.png`; those files were not on the machine used to prepare this copy,
  so the links do not resolve here.
- **Verification script.** Version 5 refers to `checks/verify_all.py` for the values it labels
  "Estimate; recomputed"; that script was not available either, and those recomputations were not re-run.
- **A PDF of 5.1.** No PDF toolchain was available; build one with the toolchain that produced the version 5 PDF.

The Markdown of version 5.1 was produced by editing the version 5 text at fixed anchors; a diff of the two
files shows 3 lines replaced and the rest added.
