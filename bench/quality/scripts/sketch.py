"""The sketch of a reference file (format of llama-perplexity --kl-divergence-base), small enough to
keep beside the results (step 3 of bench/quality): the full file holds one u16 per token of the
vocabulary per row, about 3 GB for Qwen3.6 over 48 chunks of 256 tokens.

sketch.py IN.kld OUT.npz [--k 256]

Per scored row: the exact nll (IN.kld.nll when it exists, else the one of the compressed file) and the
K most probable tokens with their log-probability, sorted (largest first; on a tie, the smaller id
first, as an argmax). cmp2.py compares a test file with it. The campaign used --k 4096."""
import sys, os, struct, numpy as np
src, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 256
f = open(src, "rb"); assert f.read(8) == b"_logits_"
n_ctx, V, C = struct.unpack("<3i", f.read(12))
toks = np.frombuffer(f.read(4 * n_ctx * C), dtype="<i4").copy()
first = n_ctx // 2; R = n_ctx - 1 - first; nv = 2 * ((V + 1) // 2) + 4
ids_all, lp_all, nllc = [], [], []
for c in range(C):
    rows = np.frombuffer(f.read(2 * R * nv), dtype="<u2").reshape(R, nv)
    hdr = rows[:, :4].copy().view("<f4")
    lp = rows[:, 4:4 + V].astype(np.float32) * hdr[:, 0:1] + hdr[:, 1:2]
    tgt = toks[c * n_ctx + first + 1:(c + 1) * n_ctx]
    nllc.append(-lp[np.arange(R), tgt])
    idx = np.argpartition(-lp, K, axis=1)[:, :K]
    v = np.take_along_axis(lp, idx, 1)
    o = np.lexsort((idx, -v), axis=1)  # on a tie, the smaller id first, as argmax
    ids_all.append(np.take_along_axis(idx, o, 1).astype(np.int32)); lp_all.append(np.take_along_axis(v, o, 1))
nllc = np.concatenate(nllc)
nll = np.fromfile(src + ".nll", dtype="<f8")[:len(nllc)] if os.path.exists(src + ".nll") else nllc
np.savez_compressed(out, n_ctx=n_ctx, vocab=V, chunks=C, tokens=toks, nll=nll, nll_compressed=nllc,
                    top_ids=np.concatenate(ids_all), top_lp=np.concatenate(lp_all), nll_exact=os.path.exists(src + ".nll"))
print(f"{out}: {C} chunks, {len(nll)} rows, K {K}, mean nll {nll.mean():.6f}, ppl {np.exp(nll.mean()):.4f}, exact {os.path.exists(src + '.nll')}")
