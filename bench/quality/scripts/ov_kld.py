"""An OpenVINO arm of the quality chain (step 4 of bench/quality, in place of an engine of arms.sh): the logits of an
optimum-intel OpenVINO IR (in the campaign, Gemma 4 E4B in int4, groups of 32: the `ov-int4g32` rows of
results/2026-11-13-quality/TABLE.md) in the format of llama-perplexity --kl-divergence-base, plus OUT.kld.nll (the exact
nll, f64). Derived from ref_tf.py, whose options it keeps (--threads and --dtype set torch's side only); compare its output
with cmp.py or cmp2.py like any arm.

ov_kld.py MODEL_DIR CORPUS.ids OUT.kld [--threads T] [--chunks N]
Needs numpy, torch, openvino and optimum-intel. The code is the one that ran (only this docstring and the comments were
translated)."""
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
t0 = time.time()
from optimum.intel import OVModelForVisualCausalLM, OVModelForCausalLM
import openvino
try:
    model = OVModelForVisualCausalLM.from_pretrained(model_dir)
except Exception as e:
    print(f"OVModelForVisualCausalLM: {str(e)[:200]}", file=sys.stderr)
    model = OVModelForCausalLM.from_pretrained(model_dir)
print(f"loaded {type(model).__name__} in {time.time()-t0:.0f} s, openvino {openvino.__version__}", file=sys.stderr)

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
        if hasattr(model, 'request') and hasattr(model.request, 'reset_state'): model.request.reset_state()
        lg = torch.as_tensor(model(input_ids=tok[None], attention_mask=torch.ones_like(tok[None])).logits)[0, first:n_ctx - 1].float()
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
# the effective n_chunk in the header
f.close(); fn.close()
print(f"[ref] done {n_chunk} chunks, {cnt} rows, mean nll {s/cnt:.6f}, ppl {np.exp(s/cnt):.4f}", file=sys.stderr)
