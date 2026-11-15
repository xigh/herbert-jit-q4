"""Step 4 of bench/quality for an engine that returns only the log-probability of the scored token and its best candidate
(vLLM, prompt_logprobs=1, vllm_nll.py): paired Δnll and top-1 agreement against the reference's sketch (sketch.py), the
fields of cmp.py with kl_mean and kl_p99 null.

cmp_nll.py REF.npz TEST OUT.json   (TEST is a prefix: TEST.nll and TEST.top1)
The code is the one that ran (only this docstring was translated): its `note` field stays in French, as in the published
out/*-vllm-bf16.json ("KL not available: prompt_logprobs does not return the full distribution")."""
import sys, json, numpy as np
r = np.load(sys.argv[1]); t = sys.argv[2]
nr = r["nll"]; nt = np.fromfile(t + ".nll", dtype="<f8"); top = np.fromfile(t + ".top1", dtype="<i4")
assert len(nr) == len(nt) == len(top), (len(nr), len(nt), len(top))
n_ctx = int(r["n_ctx"]); per = n_ctx // 2 - 1
d = nt - nr
dc = d.reshape(-1, per).mean(1)
out = {"ref": sys.argv[1], "test": t, "positions": len(nt), "ppl_ref": float(np.exp(nr.mean())), "ppl_test": float(np.exp(nt.mean())),
       "dnll": float(d.mean()), "dnll_se": float(d.std(ddof=1) / np.sqrt(len(d))), "dnll_se_chunk": float(dc.std(ddof=1) / np.sqrt(len(dc))),
       "top1": float((top == r["top_ids"][:, 0]).mean()), "kl_mean": None, "kl_p99": None, "note": "KL non disponible : prompt_logprobs ne rend pas la loi complète"}
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out))
