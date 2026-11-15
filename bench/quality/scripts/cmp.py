"""Compares a test file with the full reference file, both in the format of
llama-perplexity --kl-divergence-base (step 4 of bench/quality).

cmp.py REF.kld TEST.kld OUT.json [--test-ppl X]

The two files must hold the same tokens (checked). Per scored position: KL(ref || test) as llama.cpp
computes it (summed over the tokens with log p_ref > -16), the negative log-likelihood of the true
token on each side, and whether both put the same token first. The nll is the exact one of REF.kld.nll
and TEST.kld.nll when they exist (ref_tf.py and hkld write them), else the compressed one; for
llama.cpp, --test-ppl gives the exact perplexity printed by llama-perplexity, used for the mean
(the standard error stays the compressed one's).

OUT.json: the perplexities, the paired delta nll (test - ref) with its standard error (per position
and per chunk), the KL mean, its standard error, median, p99, p99.9 and max, and the top-1 agreement.
OUT.npz: the values per position."""
import sys, os, json, struct, numpy as np
ref_p, test_p, out = sys.argv[1:4]
def open_kld(p):
    f = open(p, "rb")
    assert f.read(8) == b"_logits_"
    n_ctx, n_vocab, n_chunk = struct.unpack("<3i", f.read(12))
    toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype="<i4")
    return f, n_ctx, n_vocab, n_chunk, toks
fr, n_ctx, V, C, tr = open_kld(ref_p)
ft, n_ctx2, V2, C2, tt = open_kld(test_p)
assert (n_ctx, V) == (n_ctx2, V2), ((n_ctx, V), (n_ctx2, V2))
C = min(C, C2)
L = n_ctx * C
assert np.array_equal(tr[:L], tt[:L]), "TOKENS DIFFER"
first = n_ctx // 2
R = n_ctx - 1 - first
nv = 2 * ((V + 1) // 2) + 4
def decode(raw):
    rows = raw.reshape(R, nv)
    hdr = rows[:, :4].copy().view("<f4")
    q = rows[:, 4:4 + V].astype(np.float32)
    return q * hdr[:, 0:1] + hdr[:, 1:2]
nll_r, nll_t, kl, same = [], [], [], []
for c in range(C):
    a = decode(np.frombuffer(fr.read(2 * R * nv), dtype="<u2"))
    b = decode(np.frombuffer(ft.read(2 * R * nv), dtype="<u2"))
    tgt = tr[c * n_ctx + first + 1:(c + 1) * n_ctx]
    nll_r.append(-a[np.arange(R), tgt]); nll_t.append(-b[np.arange(R), tgt])
    m = a > -16
    pa = np.where(m, np.exp(a), 0.0)
    kl.append((pa * np.where(m, a - b, 0.0)).sum(1, dtype=np.float64))
    same.append(a.argmax(1) == b.argmax(1))
nll_r, nll_t, kl, same = map(np.concatenate, (nll_r, nll_t, kl, same))
src = {"ref_nll": "compressed", "test_nll": "compressed"}
for name, p in (("ref", ref_p), ("test", test_p)):
    if os.path.exists(p + ".nll"):
        x = np.fromfile(p + ".nll", dtype="<f8")[:len(nll_r)]
        assert len(x) == len(nll_r)
        if name == "ref": nll_r_c, nll_r = nll_r, x
        else: nll_t = x
        src[name + "_nll"] = "exact"
d = nll_t - nll_r
n = len(d)
tp = float(sys.argv[sys.argv.index("--test-ppl") + 1]) if "--test-ppl" in sys.argv else None
dm = float(d.mean())
if tp is not None and src["test_nll"] == "compressed":
    dm = float(np.log(tp) - nll_r.mean()); src["test_nll"] = "exact ppl from llama log; se from compressed"
res = dict(ref=ref_p, test=test_p, n_ctx=n_ctx, chunks=C, positions=n, vocab=V, nll_sources=src,
           ppl_ref=float(np.exp(nll_r.mean())), ppl_test=float(np.exp(nll_r.mean() + dm)),
           nll_ref=float(nll_r.mean()), nll_test=float(nll_r.mean() + dm),
           dnll=dm, dnll_se=float(d.std(ddof=1) / np.sqrt(n)),
           dnll_se_chunk=float(d.reshape(C, R).mean(1).std(ddof=1) / np.sqrt(C)),
           kl_mean=float(kl.mean()), kl_se=float(kl.std(ddof=1) / np.sqrt(n)), kl_median=float(np.median(kl)),
           kl_p99=float(np.percentile(kl, 99)), kl_p999=float(np.percentile(kl, 99.9)), kl_max=float(kl.max()),
           top1=float(same.mean()), top1_n=int(same.sum()))
if src["ref_nll"] == "exact":
    res["ref_nll_exact_minus_compressed"] = float((nll_r - nll_r_c).mean())
json.dump(res, open(out, "w"), indent=1)
np.savez_compressed(out.replace(".json", ".npz"), nll_ref=nll_r, nll_test=nll_t, kl=kl, same=same)
print(json.dumps(res))
