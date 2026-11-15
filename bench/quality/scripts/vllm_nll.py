"""A vLLM arm of the quality chain (step 4 of bench/quality; the `vllm-bf16` rows of results/2026-11-13-quality/TABLE.md):
the log-probability of each scored token (prompt_logprobs), on the same windows as ref_tf.py, and the best candidate per
position. vLLM returns no full distribution, hence no KL: cmp_nll.py compares it with the reference's sketch.

vllm_nll.py MODEL_DIR CORPUS.ids OUT [--chunks N] -> OUT.nll (f64, as .kld.nll) and OUT.top1 (i32, vLLM's argmax)
Needs numpy and vllm. The code is the one that ran (only this docstring was translated)."""
import sys, struct, time, numpy as np
def main():
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    model_dir, ids_path, out = args[0], args[1], args[2]
    b = open(ids_path, "rb").read(); assert b[:8] == b"_ids____"
    n_ctx, n_chunk, add_bos, bos = struct.unpack("<4i", b[8:24])
    ids = np.frombuffer(b[24:24 + 4 * n_ctx * n_chunk], dtype="<i4")
    n_chunk = min(n_chunk, int(opt("--chunks", n_chunk)))
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    t0 = time.time()
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=512, language_model_only=True, enable_prefix_caching=False)
    print(f"loaded vllm {vllm.__version__} in {time.time()-t0:.0f} s", file=sys.stderr, flush=True)
    first = n_ctx // 2
    prompts = []
    for c in range(n_chunk):
        tok = ids[c * n_ctx:(c + 1) * n_ctx].astype(int).tolist()
        if add_bos: tok[0] = bos
        prompts.append(TokensPrompt(prompt_token_ids=tok))
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    t0 = time.time()
    res = llm.generate(prompts, sp)
    nll, top = [], []
    for c, r in enumerate(res):
        truth = ids[c * n_ctx:(c + 1) * n_ctx]
        for i in range(first + 1, n_ctx):
            d = r.prompt_logprobs[i]
            nll.append(-d[int(truth[i])].logprob)
            top.append(min(d.items(), key=lambda kv: kv[1].rank)[0])
    nll = np.array(nll, dtype="<f8"); nll.tofile(out + ".nll"); np.array(top, dtype="<i4").tofile(out + ".top1")
    print(f"[vllm] done {n_chunk} chunks, {len(nll)} rows, mean nll {nll.mean():.6f}, ppl {np.exp(nll.mean()):.4f} ({time.time()-t0:.0f} s)", file=sys.stderr)
if __name__ == "__main__":
    main()
