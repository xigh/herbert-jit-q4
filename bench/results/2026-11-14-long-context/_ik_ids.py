"""ik_llama.cpp ne rend pas les ids générés, seulement le texte de chaque pas (tokens_to_output_formatted_string) :
chaque morceau est ramené à son id par le vocabulaire byte-level du tokenizer HF (un id par suite d'octets),
« byte: \\xNN » pour un octet isolé. Un morceau illisible (U+FFFD) est signalé. Vérifie que les ids redonnent le texte.
_ik_ids.py MODEL_DIR IN.jsonl OUT.jsonl"""
import sys, json, os, re
from tokenizers import Tokenizer
M, IN, OUT = sys.argv[1:4]
tok = Tokenizer.from_file(os.path.join(M, "tokenizer.json"))
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
rev = {}
for t, i in tok.get_vocab(with_added_tokens=True).items():
    try:
        key = bytes(u2b[ch] for ch in t)
    except KeyError:
        key = t.encode()
    rev.setdefault(key, i)
with open(OUT, "w") as f:
    for line in open(IN):
        r = json.loads(line); ids = []; bad = []
        for k, p in enumerate(r["pieces"]):
            m = re.fullmatch(r"byte: \\x([0-9a-fA-F]{2})", p)
            key = bytes([int(m.group(1), 16)]) if m else p.encode()
            if "�" in p or key not in rev:
                bad.append(k); ids.append(-1)
            else:
                ids.append(rev[key])
        r["ids"] = ids; r["pieces_illisibles"] = bad
        ok = not bad and tok.decode([i for i in ids], skip_special_tokens=False) == (r["text"] or "") + ("<|im_end|>" if ids and ids[-1] == 248046 else "")
        r["ids_redonnent_le_texte"] = ok
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(r["id"], len(ids), "illisibles", bad, "texte_ok", ok)
