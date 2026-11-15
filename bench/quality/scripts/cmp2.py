"""Compares a test file (format of llama-perplexity --kl-divergence-base) with the
reference's sketch, sketch.py's .npz, when the full reference file is not kept (step 4 of bench/quality).

cmp2.py REF.npz TEST.kld OUT.json [--test-ppl X]

Test nll: TEST.kld.nll (exact, herbert), else --test-ppl (exact, from llama-perplexity's log) for the
mean, and the compressed file for the standard error. KL(ref || test) over the sketch's K tokens plus
the remaining mass lumped together on each side: the KL of the coarsened distribution, a lower bound
of the full KL (the gap was measured as zero on the test). Top-1: the argmax on each side. OUT.json
has the fields of cmp.py's, OUT.npz the values per position."""
import sys, os, json, struct, numpy as np
ref, test, out = sys.argv[1:4]
tp = float(sys.argv[sys.argv.index("--test-ppl") + 1]) if "--test-ppl" in sys.argv else None
r = np.load(ref)
f = open(test, "rb"); assert f.read(8) == b"_logits_"
n_ctx, V, C = struct.unpack("<3i", f.read(12))
toks = np.frombuffer(f.read(4 * n_ctx * C), dtype="<i4")
assert (n_ctx, V) == (int(r["n_ctx"]), int(r["vocab"])), ((n_ctx, V), (int(r["n_ctx"]), int(r["vocab"])))
C = min(C, int(r["chunks"])); L = n_ctx * C
if not np.array_equal(toks[:L], r["tokens"][:L]):
    d = int(np.argmax(toks[:L] != r["tokens"][:L])); raise SystemExit(f"TOKENS DIFFER at {d}: test {toks[d-3:d+3]} ref {r['tokens'][d-3:d+3]}")
first = n_ctx // 2; R = n_ctx - 1 - first; nv = 2 * ((V + 1) // 2) + 4
N = C * R
top_ids, top_lp, nll_r = r["top_ids"][:N], r["top_lp"][:N].astype(np.float64), r["nll"][:N]
nll_tc, kl, same = [], [], []
for c in range(C):
    rows = np.frombuffer(f.read(2 * R * nv), dtype="<u2").reshape(R, nv)
    hdr = rows[:, :4].copy().view("<f4")
    lq = rows[:, 4:4 + V].astype(np.float32) * hdr[:, 0:1] + hdr[:, 1:2]
    tgt = toks[c * n_ctx + first + 1:(c + 1) * n_ctx]
    nll_tc.append(-lq[np.arange(R), tgt].astype(np.float64))
    sl = slice(c * R, (c + 1) * R)
    lp = top_lp[sl]; ids = top_ids[sl]
    lqs = np.take_along_axis(lq, ids, 1).astype(np.float64)
    p = np.exp(lp); P = p.sum(1); Q = np.exp(lqs).sum(1)
    k = (p * (lp - lqs)).sum(1)
    rest = np.clip(1 - P, 0, None); qrest = np.clip(1 - Q, 1e-30, None)
    k += np.where(rest > 1e-12, rest * (np.log(np.maximum(rest, 1e-30)) - np.log(qrest)), 0.0)
    kl.append(k); same.append(lq.argmax(1) == ids[:, 0])
nll_tc, kl, same = map(np.concatenate, (nll_tc, kl, same))
src = "compressed"
nll_t = nll_tc
if os.path.exists(test + ".nll"):
    nll_t = np.fromfile(test + ".nll", dtype="<f8")[:N]; assert len(nll_t) == N; src = "exact (.nll)"
d = nll_t - nll_r
n = len(d)
dm = d.mean()
if tp is not None and src == "compressed":
    dm = float(np.log(tp) - nll_r.mean()); src = "exact ppl from llama log; se from compressed"
res = dict(ref=ref, test=test, n_ctx=n_ctx, chunks=C, positions=n, vocab=V, test_nll=src, ref_nll_exact=bool(r["nll_exact"]),
           ppl_ref=float(np.exp(nll_r.mean())), ppl_test=float(np.exp(nll_r.mean() + dm)),
           dnll=float(dm), dnll_se=float(d.std(ddof=1) / np.sqrt(n)), dnll_se_chunk=float(d.reshape(C, R).mean(1).std(ddof=1) / np.sqrt(C)),
           kl_mean=float(kl.mean()), kl_se=float(kl.std(ddof=1) / np.sqrt(n)), kl_median=float(np.median(kl)),
           kl_p99=float(np.percentile(kl, 99)), kl_p999=float(np.percentile(kl, 99.9)), kl_max=float(kl.max()),
           top1=float(same.mean()), nonfinite_test=int((~np.isfinite(nll_tc)).sum()))
json.dump(res, open(out, "w"), indent=1)
np.savez_compressed(out[:-5] + ".npz", nll_ref=nll_r, nll_test=nll_t, kl=kl, same=same)
print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items() if k not in ("ref", "test")}))
