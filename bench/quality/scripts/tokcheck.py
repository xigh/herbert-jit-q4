"""Checks that the Hugging Face tokenizer (what ids.py uses for every engine) and
llama.cpp's (llama-tokenize on the GGUF) give the same ids over the first N tokens of a corpus.

tokcheck.py HF_DIR GGUF CORPUS ADD_BOS BOS N LLAMA_TOKENIZE

ADD_BOS and BOS as for ids.py; LLAMA_TOKENIZE is the path of llama.cpp's llama-tokenize binary.
Prints one line: both lengths, and identical or the first difference."""
import sys, subprocess, ast
from tokenizers import Tokenizer
M, G, T, ab, bos, N, LT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
text = open(T, encoding="utf-8").read()
a = Tokenizer.from_file(M + "/tokenizer.json").encode(text, add_special_tokens=False).ids
if ab: a = [bos] + a
out = subprocess.run([LT, "-m", G, "-f", T, "--ids", "--log-disable"], capture_output=True, text=True).stdout
b = ast.literal_eval(out.strip().splitlines()[-1])
n = min(N, len(a), len(b))
d = next((i for i in range(n) if a[i] != b[i]), None)
print(f"TOKCHECK {G.split('/')[-1]} {T.split('/')[-1]}: hf {len(a)} llama {len(b)} first {n} identical: {d is None}" + ("" if d is None else f" first diff at {d}: hf {a[d-2:d+3]} llama {b[d-2:d+3]}"))
