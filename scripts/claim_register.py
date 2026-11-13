# SPDX-License-Identifier: AGPL-3.0-only
"""Render docs/research/claim-register.md from docs/research/claim-register.json.

The JSON is the register; edit it, then run `py -I scripts/claim_register.py` (or
`python3 scripts/claim_register.py`). `--check` fails when the Markdown is out of date.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "research" / "claim-register.json"
DST = ROOT / "docs" / "research" / "claim-register.md"


def cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def render(doc):
    claims = doc["claims"]
    counts = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    lines = [
        "# THME claim register",
        "",
        f"**Source:** [`{pathlib.Path(doc['source']).name}`]({pathlib.Path(doc['source']).name}), "
        f"consolidated {doc['extracted']} (THME-3000)",
        "**Evidence:** status labels are the dossier author's own classification, copied as written. "
        "Nothing here was re-measured, and no literature citation in the dossier has been checked "
        "against its source: every citation is **unverified**.",
        "**Generated** from [`claim-register.json`](claim-register.json) by "
        "`scripts/claim_register.py`; edit the JSON, not this file.",
        "",
        "## Status labels",
        "",
        "| Label | Meaning | Claims |",
        "| :--- | :--- | ---: |",
    ]
    for label, meaning in doc["labels"].items():
        lines.append(f"| {cell(label)} | {cell(meaning)} | {counts.get(label, 0)} |")
    lines += [
        "",
        "*Qualified* marks a status the dossier limits to a case, for example \"S for independent "
        "noise; C for state error\"; the qualifier opens the note. *Page* is the PDF page where the "
        "claim's register table starts. *Section* is the paper section for Appendix A rows and the "
        "appendix for the others.",
        "",
        "## Register",
        "",
        "| ID | Claim | Status | Section | Page | Note |",
        "| :--- | :--- | :--- | :--- | ---: | :--- |",
    ]
    for c in claims:
        status = c["status"] + (" (qualified)" if c["status_qualified"] else "")
        lines.append(
            f"| {c['id']} | {cell(c['claim'])} | {cell(status)} | {cell(c['section'])} | {c['page']} | {cell(c['note'])} |"
        )
    return "\n".join(lines) + "\n"


def main():
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    ids = [c["id"] for c in doc["claims"]]
    if len(ids) != len(set(ids)):
        sys.exit("claim-register.json: duplicate claim IDs")
    for c in doc["claims"]:
        if c["status"] not in doc["labels"]:
            sys.exit(f"claim-register.json: {c['id']} has unknown status {c['status']!r}")
    text = render(doc)
    if "--check" in sys.argv:
        if DST.read_text(encoding="utf-8") != text:
            sys.exit("claim-register.md is out of date; run scripts/claim_register.py")
        print(f"claim register up to date ({len(ids)} claims)")
        return
    DST.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {DST.relative_to(ROOT)} ({len(ids)} claims)")


if __name__ == "__main__":
    main()
