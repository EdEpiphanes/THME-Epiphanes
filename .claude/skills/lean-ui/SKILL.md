---
name: lean-ui
description: Token-efficient UI work: pages, components, dashboards, artifacts, GUI screens, charts, and visual polish. Use whenever building or changing anything visual (HTML, CSS, React, egui, layouts, themes, screenshots). Builds from one short spec and shared design tokens, reuses artifact-design and dataviz, and verifies with one cheap screenshot per viewport.
---

# Lean UI

Spec once, build from tokens, verify cheaply. Sonnet builds; haiku drafts copy and labels; consult the opus advisor only for ambiguous layout or information-architecture decisions.

1. **Spec in 5 lines**: purpose, user, the 3 key states (empty, loaded, error), viewports (phone 360px and desktop), themes (light and dark).
2. **Reuse before writing**: `Grep` for existing components and styles in the project. For pages and artifacts, load `artifact-design` once; for any chart, load `dataviz` once. Do not improvise what those skills already specify.
3. **Tokens first**: define colors, spacing, and type scale as CSS variables (or constants) in one place. Components use tokens only, with no hardcoded colors. Dark mode then costs almost nothing.
4. **Write once**: `Write` the new file in one pass; use `Edit` for tweaks. Do not regenerate a whole file to change a style. Use inline SVG for icons and no extra libraries unless needed (pinned versions from cdnjs).
5. **Verify with one shot per viewport**: Playwright is pre-installed (`chromium.launch()` needs no options). Take screenshots at `deviceScaleFactor: 1` for your own checks and 2 only for ones the user will see. Check overflow, contrast, and the empty and error states. Fix everything found in one batch, then re-shoot once.
6. **Floor for quality**: text contrast of at least 4.5:1, visible focus, tap targets of at least 44px, no meaning carried by color alone, no horizontal scroll on phones.

Defects to avoid: hardcoded colors, missing empty state, large inline data, multiple screenshot-fix loops for issues one pass would catch.
