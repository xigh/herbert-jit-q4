"""The tokens of a corpus, cut into chunks, for hkld and ref_tf.py (step 1 of bench/quality).

ids.py MODEL_DIR CORPUS N_CTX N_CHUNK ADD_BOS BOS OUT.ids

The corpus is tokenized with the model's Hugging Face tokenizer (MODEL_DIR/tokenizer.json, `tokenizers`
package), without special tokens; BOS (its id) goes first when ADD_BOS is 1, as llama-perplexity does
(common_tokenize(..., add_special=true)): give what the GGUF says (tokenizer.ggml.add_bos_token,
tokenizer.ggml.bos_token_id); tokcheck.py checks that both tokenizers agree. The first
N_CTX * N_CHUNK tokens are written to OUT.ids: "_ids____", four little-endian i32 (n_ctx, n_chunk,
add_bos, bos), then the tokens as i32. OUT.ids.all: every token of the corpus, as text. The campaign
used N_CTX 256, 48 chunks of wikitext-2 and 24 of Les Miserables."""
import sys, struct
from tokenizers import Tokenizer
model, corpus, n_ctx, n_chunk, add_bos, bos, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
tok = Tokenizer.from_file(model + "/tokenizer.json")
text = open(corpus, encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=False).ids
# as llama-perplexity does (common_tokenize(..., add_special=true)): BOS at the start when the model uses one
if add_bos:
    ids = [bos] + ids
need = n_ctx * n_chunk
print(f"{corpus}: {len(ids)} tokens, need {need}", file=sys.stderr)
assert len(ids) >= need
with open(out, "wb") as f:
    f.write(b"_ids____")
    f.write(struct.pack("<4i", n_ctx, n_chunk, add_bos, bos))
    f.write(struct.pack(f"<{need}i", *ids[:need]))
open(out + ".all", "w").write(" ".join(map(str, ids)) + "\n")
