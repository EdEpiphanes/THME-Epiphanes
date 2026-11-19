# SPDX-License-Identifier: AGPL-3.0-only
"""Third-party code register check.

    python scripts/third_party.py --check   # CI: every noticed file has a row
    python scripts/third_party.py --self-test

Checks that every tracked source file whose header names a non-AGPL licence has a row in
docs/third-party.md (backticked tokens in the Files column).
"""
import subprocess
import sys

LICENSE = "AGPL-3.0-only"
EXTENSIONS = {".rs", ".ts", ".js", ".py", ".sh", ".ps1", ".css"}
COMMENT = {
    ".rs": "//",
    ".ts": "//",
    ".js": "//",
    ".py": "#",
    ".sh": "#",
    ".ps1": "#",
    ".css": "/*",
}

# shortcut: only files whose header names a non-AGPL licence are detected;
# code copied without a header is not, upgrade when copies appear


def tracked():
    """Yield (path, extension) for every tracked source file."""
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    for path in out.splitlines():
        ext = path[path.rfind("."):] if "." in path else ""
        if ext in COMMENT:
            yield path, ext


def notice_expression(first_lines):
    """Return the SPDX expression if the file carries a third-party licence, else None."""
    for line in first_lines:
        marker = "SPDX-License-Identifier:"
        if marker in line:
            expr = line.split(marker, 1)[1]
            expr = expr.replace("*/", "").strip().strip("*").strip()
            if expr == LICENSE:
                return None
            return expr or None
    return None


def listed_paths(doc_text):
    """Return the set of backticked tokens in the register document."""
    return set(__import__("re").findall(r"`([^`]+)`", doc_text))


def _self_test():
    assert notice_expression(
        ["// SPDX-License-Identifier: AGPL-3.0-only AND MIT", "// x"]
    ) == "AGPL-3.0-only AND MIT"
    assert notice_expression(
        ["/* SPDX-License-Identifier: MIT", " From the kit */"]
    ) == "MIT"
    assert notice_expression(["// SPDX-License-Identifier: AGPL-3.0-only"]) is None
    assert notice_expression(["fn main() {}"]) is None
    assert listed_paths(
        "| a | b | c | d | e | `src/x.rs`, `y.css` |)"
    ) == {"src/x.rs", "y.css"}
    print("ok")


def check():
    notices = []
    for path, ext in tracked():
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        body = text.lstrip("﻿")
        lines = body.split("\n")[:3]
        expr = notice_expression(lines)
        if expr is not None:
            notices.append(path)
    try:
        with open("docs/third-party.md", encoding="utf-8") as f:
            doc = f.read()
    except FileNotFoundError:
        print("::error file=docs/third-party.md::docs/third-party.md not found")
        return 1
    listed = listed_paths(doc)
    missing = [p for p in notices if p not in listed]
    if missing:
        for path in missing:
            print(
                f"::error file={path}::third-party notice without a row in docs/third-party.md"
            )
        return 1
    return 0


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--check", "--self-test"):
        print(__doc__)
        return 2
    if sys.argv[1] == "--self-test":
        _self_test()
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
