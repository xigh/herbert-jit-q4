#!/usr/bin/env python3
"""dump: writes a GGUF's header (the bytes before the data) and layout (name, offset, size of each tensor).
build: rebuilds a GGUF with that header and layout, taking each tensor's data from another GGUF (in the campaign, the
layout dumped on the Ryzen 9 9900X, rebuilt on the EPYC 4344P from a convert of another day).
Usage: layout.py dump GGUF HEADER.bin LAYOUT.tsv LLAMA_DIR ; layout.py build HEADER.bin LAYOUT.tsv SOURCE.gguf OUT.gguf SIZE LLAMA_DIR"""
import sys, os
mode = sys.argv[1]
sys.path.insert(0, os.path.join(sys.argv[-1], "gguf-py")); import gguf
if mode == "dump":
    g, hb, lt = sys.argv[2:5]
    r = gguf.GGUFReader(g); start = min(int(t.data_offset) for t in r.tensors)
    open(hb, "wb").write(open(g, "rb").read(start))
    with open(lt, "w") as f:
        for t in r.tensors: f.write(f"{t.name}\t{int(t.data_offset)}\t{int(t.n_bytes)}\n")
else:
    hb, lt, src, out, size = sys.argv[2:7]
    r = gguf.GGUFReader(src); S = {t.name: (int(t.data_offset), int(t.n_bytes)) for t in r.tensors}
    fi = open(src, "rb")
    with open(out, "wb") as fo:
        fo.write(open(hb, "rb").read())
        for l in open(lt):
            n, off, nb = l.split("\t"); off, nb = int(off), int(nb)
            so, sn = S[n]; assert sn == nb, n
            fo.seek(off); fi.seek(so); fo.write(fi.read(nb))
        fo.truncate(int(size))
    print("wrote", out)
