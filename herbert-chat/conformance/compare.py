#!/usr/bin/env python3
"""The two renders side by side: per template, how many are byte-identical, and for each difference where it starts,
with the text around it (repr, to show spaces and newlines). Exits with an error if anything differs.

usage: python3 -I compare.py REFERENCE_JSON OURS_JSON [REPORT_TXT]
"""
import json, sys
from collections import defaultdict

ref = json.load(open(sys.argv[1], encoding="utf-8"))
ours = json.load(open(sys.argv[2], encoding="utf-8"))
out = open(sys.argv[3], "w", encoding="utf-8") if len(sys.argv) > 3 else sys.stdout
versions = ref.pop("_versions", {})
print(f"reference: transformers {versions.get('transformers')}, jinja2 {versions.get('jinja2')}", file=out)
per = defaultdict(lambda: [0, 0])
diffs = []
for key, r in ref.items():
    o = ours.get(key) or ours.get(key.split("|")[0] + "|*")
    name = key.split("|")[0]
    per[name][1] += 1
    if o is None:
        diffs.append((key, "missing from our render", ""))
        continue
    if "error" in r or "error" in o:
        # Both refuse: the same verdict (the messages differ: Python's and minijinja's).
        if "error" in r and "error" in o:
            per[name][0] += 1
        else:
            diffs.append((key, f"reference: {r.get('error', 'render')}; ours: {o.get('error', 'render')}", ""))
        continue
    a, b = r["ok"], o["ok"]
    if a == b:
        per[name][0] += 1
        continue
    i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    ctx = f"    reference: {a[max(0, i - 60):i + 80]!r}\n    ours     : {b[max(0, i - 60):i + 80]!r}"
    diffs.append((key, f"differs at byte {i} (lengths {len(a)} and {len(b)})", ctx))
for name, (same, total) in per.items():
    print(f"{name:22} {same:3}/{total} identical", file=out)
print(f"total {sum(s for s, _ in per.values())}/{sum(t for _, t in per.values())}", file=out)
for key, what, ctx in diffs:
    print(f"\n{key}: {what}", file=out)
    if ctx:
        print(ctx, file=out)
if out is not sys.stdout:
    out.close()
    print(open(sys.argv[3], encoding="utf-8").read())
sys.exit(1 if diffs else 0)
