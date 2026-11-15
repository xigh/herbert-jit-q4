"""The table of a quality campaign (step 5 of bench/quality).

table.py DIR > TABLE.md

DIR holds one directory per machine, each with the out/NAME-CORPUS-ARM.json that cmp.py or cmp2.py
wrote; one row per (model, corpus, arm, machine): reference and test perplexity, delta nll with its
standard error, mean KL, KL p99, top-1 agreement. results/2026-11-13-quality/TABLE.md is its output."""
import json, glob, os, re, sys
W = sys.argv[1]


def machine(folder):
    """ryzen-9-9900x -> Ryzen 9 9900X, epyc-4344p -> EPYC 4344P."""
    return " ".join("EPYC" if t == "epyc" else t.upper() if any(c.isdigit() for c in t) else t.capitalize()
                    for t in folder.split("-"))


rows = []
for f in sorted(glob.glob(W + "/*/out/*.json")):
    host = f[len(W) + 1:].split("/")[0]
    n = os.path.basename(f)[:-5]
    if n.endswith("-sketch"):
        continue
    d = json.load(open(f))
    # The model's own name may hold dashes (q3vl-instruct): the corpus is the anchor.
    m, k, arm = re.fullmatch(r"(.+?)-(wiki|fr)-(.+)", n).groups()
    rows.append((m, k, arm, machine(host), d))
print("| model | corpus | arm | machine | ref ppl | ppl | Δnll ± se | mean KL | KL p99 | top-1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m, k, arm, host, d in sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3])):
    g = lambda key, fmt: "—" if d.get(key) is None else format(d[key] * (100 if key == "top1" else 1), fmt)
    print(f"| {m} | {k} | {arm} | {host} | {g('ppl_ref', '.3f')} | {g('ppl_test', '.3f')} | {g('dnll', '+.4f')} ± {g('dnll_se', '.4f')} | {g('kl_mean', '.4f')} | {g('kl_p99', '.3f')} | {g('top1', '.1f')} % |")
