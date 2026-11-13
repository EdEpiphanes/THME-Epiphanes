#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Mirror the active backlog (BACKLOG.md, between the backlog:start/end markers) to GitHub.

Creates any missing labels, milestones and issues. Issues are matched by their ticket ID
prefix ("[THME-2201] ..."), so re-running never duplicates. Existing issues are not edited.

Default is a dry run that only prints the plan and needs no network or token.
  python scripts/sync_github_backlog.py
  GITHUB_TOKEN=... python scripts/sync_github_backlog.py --apply [--repo owner/name]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

LABELS = {
    "type:bug": "d73a4a", "type:feature": "1f6feb", "type:experiment": "8250df",
    "type:chore": "6e7781", "type:docs": "0e8a16",
    "priority:p0": "b60205", "priority:p1": "d93f0b", "priority:p2": "fbca04",
    "status:triage": "ededed", "status:ready": "0e8a16", "status:blocked": "000000",
}
AREA_COLOR = "c5def5"
TICKET = re.compile(r"^### \[(THME-\d+)\] (.+)$")
FIELD = re.compile(r"\*\*(\w[\w ]*)\*\*:\s*([^·]+)")


def parse_backlog(path):
    text = open(path, encoding="utf-8").read()
    start, end = text.find("<!-- backlog:start -->"), text.find("<!-- backlog:end -->")
    if start < 0 or end < start:
        raise SystemExit("error: backlog markers not found in BACKLOG.md")
    tickets, current = [], None
    for line in text[start:end].splitlines():
        match = TICKET.match(line)
        if match:
            current = {"id": match.group(1), "title": match.group(2).strip(), "lines": []}
            tickets.append(current)
            continue
        if current is None:
            continue
        if line.startswith("## "):
            current = None
            continue
        current["lines"].append(line)
        if line.startswith("- **Milestone**"):
            current.update({k.strip().lower(): v.strip() for k, v in FIELD.findall(line)})
    return tickets


def labels_for(ticket):
    labels = [f"type:{ticket.get('type', 'feature')}", f"priority:{ticket.get('priority', 'p2')}",
              f"status:{ticket.get('status', 'triage')}"]
    if ticket.get("area"):
        labels.append(f"area:{ticket['area']}")
    return labels


def default_repo():
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?\s*$", url.strip())
    return match.group(1) if match else None


class GitHub:
    def __init__(self, repo, token):
        self.base = f"https://api.github.com/repos/{repo}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                        "User-Agent": "thme-backlog-sync"}

    def call(self, method, path, body=None):
        request = urllib.request.Request(self.base + path, method=method, headers=self.headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read() or b"null")

    def all(self, path):
        items, page = [], 1
        while True:
            batch = self.call("GET", f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
            items += batch
            if len(batch) < 100:
                return items
            page += 1


def main():
    parser = argparse.ArgumentParser(description="Mirror BACKLOG.md to GitHub labels, milestones and issues")
    parser.add_argument("--backlog", default="BACKLOG.md")
    parser.add_argument("--repo", default=None, help="owner/name (default: from git remote origin)")
    parser.add_argument("--apply", action="store_true", help="create missing items (needs GITHUB_TOKEN)")
    args = parser.parse_args()

    tickets = parse_backlog(args.backlog)
    wanted_labels = dict(LABELS)
    for t in tickets:
        for label in labels_for(t):
            wanted_labels.setdefault(label, AREA_COLOR)
    milestones = sorted({t["milestone"] for t in tickets if t.get("milestone")})

    if not args.apply:
        print(f"Dry run: {len(tickets)} tickets, {len(wanted_labels)} labels, {len(milestones)} milestones")
        print("Milestones:", ", ".join(milestones))
        for t in tickets:
            print(f"  [{t['id']}] {t['title']}  ({t.get('milestone', '-')}; {', '.join(labels_for(t))})")
        print("Re-run with --apply and GITHUB_TOKEN set to create the missing items.")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    repo = args.repo or default_repo()
    if not token or not repo:
        print("error: --apply needs GITHUB_TOKEN and a repository (--repo owner/name)", file=sys.stderr)
        return 1
    gh = GitHub(repo, token)
    existing_labels = {l["name"] for l in gh.all("/labels")}
    for name, color in wanted_labels.items():
        if name not in existing_labels:
            gh.call("POST", "/labels", {"name": name, "color": color})
            print(f"created label {name}")
    existing_milestones = {m["title"]: m["number"] for m in gh.all("/milestones?state=all")}
    for title in milestones:
        if title not in existing_milestones:
            existing_milestones[title] = gh.call("POST", "/milestones", {"title": title})["number"]
            print(f"created milestone {title}")
    existing_ids = set()
    for issue in gh.all("/issues?state=all"):
        match = re.match(r"\[(THME-\d+)\]", issue["title"])
        if match:
            existing_ids.add(match.group(1))
    for t in tickets:
        if t["id"] in existing_ids:
            print(f"exists  [{t['id']}]")
            continue
        body = "\n".join(t["lines"]).strip() + "\n\n_Source: BACKLOG.md. Edit the backlog, not only this issue._"
        payload = {"title": f"[{t['id']}] {t['title']}", "body": body, "labels": labels_for(t)}
        if t.get("milestone") in existing_milestones:
            payload["milestone"] = existing_milestones[t["milestone"]]
        created = gh.call("POST", "/issues", payload)
        print(f"created [{t['id']}] -> #{created['number']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
