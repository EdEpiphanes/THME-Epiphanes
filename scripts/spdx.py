# SPDX-License-Identifier: AGPL-3.0-only
"""Check or add the SPDX licence header on every tracked source file.

    python scripts/spdx.py --check   # CI: list files without the header, exit 1 if any
    python scripts/spdx.py --fix     # add the header where it is missing

The header goes on the first line, or right after a shebang line.
"""
import subprocess
import sys

LICENSE = "AGPL-3.0-only"
COMMENT = {".rs": "//", ".ts": "//", ".js": "//", ".py": "#", ".sh": "#", ".ps1": "#"}


def header(ext):
    return f"{COMMENT[ext]} SPDX-License-Identifier: {LICENSE}"


def tracked():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    for path in out.splitlines():
        ext = path[path.rfind("."):] if "." in path else ""
        if ext in COMMENT:
            yield path, ext


def has_header(lines):
    return any("SPDX-License-Identifier:" in line for line in lines[:3])


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--check", "--fix"):
        print(__doc__)
        return 2
    fix = sys.argv[1] == "--fix"
    missing = []
    for path, ext in tracked():
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        bom = text.startswith("﻿")
        body = text[1:] if bom else text
        lines = body.split("\n")
        if has_header(lines):
            continue
        missing.append(path)
        if fix:
            newline = "\r\n" if "\r\n" in body else "\n"
            at = 1 if lines and lines[0].startswith("#!") else 0
            lines.insert(at, header(ext) + ("\r" if newline == "\r\n" else ""))
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(("﻿" if bom else "") + "\n".join(lines))
    if fix:
        print(f"added the SPDX header to {len(missing)} files")
        return 0
    for path in missing:
        print(f"::error file={path}::missing 'SPDX-License-Identifier: {LICENSE}' header")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
