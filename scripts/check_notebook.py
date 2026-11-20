# SPDX-License-Identifier: AGPL-3.0-only
"""Check the example notebooks: nbformat 4, no stored outputs, no absolute user paths.

    python scripts/check_notebook.py                 # every examples/notebooks/*.ipynb
    python scripts/check_notebook.py FILE.ipynb ...  # the files given

Exit 0 when every notebook passes, 1 when any check fails, 2 when no notebook was found.
Standard library only.
"""
import glob
import json
import re
import sys

DEFAULT_GLOB = "examples/notebooks/*.ipynb"

# The same shapes the CI hygiene job rejects: a Windows profile, /home/<user> and macOS /Users/<user>.
USER_PATH = re.compile(r"[A-Za-z]:\\+Users\\+[^\\\s\"]+|/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|/c/Users/")


def problems(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        nb = json.loads(text)
    except json.JSONDecodeError as err:
        return [f"not valid JSON: {err}"]
    found = []
    if not isinstance(nb, dict):
        return ["top level is not a JSON object"]
    if nb.get("nbformat") != 4:
        found.append(f"nbformat is {nb.get('nbformat')!r}, expected 4")
    if not isinstance(nb.get("nbformat_minor"), int) or nb["nbformat_minor"] < 5:
        found.append("nbformat_minor must be an integer of at least 5 (cells need ids)")
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        found.append("no cells")
        cells = []
    for index, cell in enumerate(cells):
        kind = cell.get("cell_type") if isinstance(cell, dict) else None
        if kind not in ("markdown", "code", "raw"):
            found.append(f"cell {index}: unknown cell_type {kind!r}")
            continue
        if not isinstance(cell.get("id"), str) or not cell["id"]:
            found.append(f"cell {index}: missing id")
        if kind == "code":
            if cell.get("outputs") != []:
                found.append(f"cell {index}: has stored outputs")
            if cell.get("execution_count") is not None:
                found.append(f"cell {index}: has an execution_count")
    for match in USER_PATH.finditer(text):
        found.append(f"absolute user path: {match.group(0)!r}")
    return found


def main(argv):
    paths = argv or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print(f"no notebook found for {DEFAULT_GLOB}")
        return 2
    failed = False
    for path in paths:
        for problem in problems(path):
            failed = True
            print(f"::error file={path}::{problem}")
    if not failed:
        print(f"{len(paths)} notebook(s) checked: nbformat 4, no stored outputs, no user paths")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
