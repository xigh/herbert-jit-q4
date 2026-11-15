"""The reference: transformers in bf16 (or f32) on the tokens of ids.py, in the format of
llama-perplexity --kl-divergence-base (step 2 of bench/quality).

ref_tf.py MODEL_DIR IDS OUT.kld [--threads T] [--chunks N] [--dtype bf16|f32] [--attn IMPL]

Each chunk of IDS is read in one forward pass (no cache), BOS put first when the file says so, and the
positions n_ctx/2 .. n_ctx-2 are scored, as hkld and llama-perplexity score them. OUT.kld: the same
header and the same compressed log-softmax as llama.cpp (see the header of hkld). OUT.kld.nll: the
exact negative log-likelihood of each scored row (f64). --attn passes attn_implementation to
transformers; --threads sets torch's threads (default 8). Needs torch, transformers and numpy; the
model is loaded with AutoModelForCausalLM, else AutoModelForImageTextToText. The full file is large
(one u16 per token of the vocabulary per row: about 3 GB for Qwen3.6 over 48 chunks of 256 tokens):
sketch.py keeps what the comparison needs."""
import sys, struct, time, numpy as np, torch
args = sys.argv[1:]
def opt(k, d):
    return args[args.index(k) + 1] if k in args else d
model_dir, ids_path, out = args[0], args[1], args[2]
torch.set_num_threads(int(opt("--threads", "8")))
dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[opt("--dtype", "bf16")]
b = open(ids_path, "rb").read()
assert b[:8] == b"_ids____"
n_ctx, n_chunk, add_bos, bos = struct.unpack("<4i", b[8:24])
ids = np.frombuffer(b[24:24 + 4 * n_ctx * n_chunk], dtype="<i4")
n_chunk = min(n_chunk, int(opt("--chunks", n_chunk)))
import transformers
t0 = time.time()
model = None
for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
    try:
        kw = {'attn_implementation': opt('--attn', None)} if '--attn' in args else {}
        model = getattr(transformers, cls).from_pretrained(model_dir, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True, **kw)
        break
    except Exception as e:
        print(f"{cls}: {str(e)[:200]}", file=sys.stderr)
if model is None:
    sys.exit(f"{model_dir}: neither class loads it")
model.eval()
print(f"loaded {type(model).__name__} in {time.time()-t0:.0f} s, transformers {transformers.__version__}, torch {torch.__version__}, dtype {dtype}", file=sys.stderr)
first = n_ctx // 2
f = open(out, "wb")
fn = open(out + ".nll", "wb")
vocab = None
s, cnt = 0.0, 0
t0 = time.time()
for c in range(n_chunk):
    tok = torch.tensor(ids[c * n_ctx:(c + 1) * n_ctx].astype(np.int64))
    truth = tok.clone()
    if add_bos:
        tok[0] = bos
    with torch.inference_mode():
        lg = model(input_ids=tok[None], use_cache=False).logits[0, first:n_ctx - 1].float()
    if vocab is None:
        vocab = lg.shape[1]
        f.write(b"_logits_" + struct.pack("<3i", n_ctx, vocab, n_chunk))
        f.write(ids[:n_ctx * n_chunk].astype("<i4").tobytes())
    tgt = truth[first + 1:n_ctx]
    # exact nll
    l64 = lg.double()
    nll = (torch.logsumexp(l64, 1) - l64.gather(1, tgt[:, None])[:, 0]).numpy()
    assert np.isfinite(nll).all()
    fn.write(nll.astype("<f8").tobytes())
    s += nll.sum(); cnt += len(nll)
    # llama.cpp's compression (log_softmax of perplexity.cpp)
    mx = lg.max(1).values
    mn = torch.maximum(lg.min(1).values, mx - 16)
    sum_exp = torch.exp(lg - mx[:, None]).double().sum(1)
    lse = torch.log(sum_exp).float()
    min_log_prob = mn - mx - lse
    scale = (mx - mn) / 65535.0
    inv = 1.0 / scale
    q = torch.where(lg > mn[:, None], torch.round(inv[:, None] * (lg - mn[:, None])), torch.zeros_like(lg)).to(torch.int32).numpy().astype("<u2")
    nv = 2 * ((vocab + 1) // 2) + 4
    rows = np.zeros((lg.shape[0], nv), dtype="<u2")
    hdr = np.stack([scale.numpy(), min_log_prob.numpy()], 1).astype("<f4").view("<u2")
    rows[:, :4] = hdr
    rows[:, 4:4 + vocab] = q
    f.write(rows.tobytes())
    print(f"[ref] chunk {c+1}/{n_chunk} ppl {np.exp(s/cnt):.4f} ({time.time()-t0:.0f} s)", file=sys.stderr, flush=True)
# the header holds the n_chunk actually written
f.close(); fn.close()
print(f"[ref] done {n_chunk} chunks, {cnt} rows, mean nll {s/cnt:.6f}, ppl {np.exp(s/cnt):.4f}", file=sys.stderr)
