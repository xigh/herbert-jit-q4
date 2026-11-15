#!/usr/bin/env python3
"""Rebuilds in place the Q4_0 tensors of a skeleton GGUF (a pure Q4_0 file's header and layout, from layout.py, Q4_0 data
zeroed) from the safetensors, with no bf16 GGUF on disk: llama.cpp 5676dfbe0's convert_hf_to_gguf.py makes each tensor in
memory, a hooked GGUFWriter.add_tensor widens it to f32 (bf16 << 16, as llama-quantize does), quantizes it with libggml-base's
ggml_quantize_chunk and writes it at its offset (or compares, --check). Accept the result only if the whole file's sha256
matches llama-quantize's. Usage: rebuild_q40.py LLAMA_DIR HF_DIR TARGET.gguf [--check] [--only REGEX] [--threads N]"""
import argparse, ctypes, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("llama"); ap.add_argument("hf"); ap.add_argument("target")
ap.add_argument("--check", action="store_true"); ap.add_argument("--only", default=None); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.llama, "gguf-py")); sys.path.insert(0, a.llama)
import gguf

lib = ctypes.CDLL(os.path.join(a.llama, "build", "bin", "libggml-base.so"))
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
Q4_0 = int(gguf.GGMLQuantizationType.Q4_0)
lib.ggml_quantize_init(Q4_0)

rd = gguf.GGUFReader(a.target)
lay = {t.name: (int(t.data_offset), int(t.n_bytes), t.tensor_type, [int(x) for x in t.shape]) for t in rd.tensors}
del rd
fd = os.open(a.target, os.O_RDONLY if a.check else os.O_RDWR)
pool = ThreadPoolExecutor(a.threads)
stats = {"q4": 0, "ok": 0, "diff": 0, "other_same": 0, "other_diff": 0, "missing": 0, "bytes": 0}
seen = set()
t0 = time.time()


def quant(f32, n_per_row, nbytes):
    nrows = f32.size // n_per_row
    dst = np.zeros(nbytes, np.uint8)
    step = max(1, nrows // (a.threads * 4))
    def job(r0):
        r1 = min(nrows, r0 + step)
        lib.ggml_quantize_chunk(Q4_0, f32.ctypes.data, dst.ctypes.data, r0 * n_per_row, r1 - r0, n_per_row, None)
    list(pool.map(job, range(0, nrows, step)))
    return dst


def hook(self, name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
    if a.only and not re.search(a.only, name):
        return
    seen.add(name)
    if name not in lay:
        stats["missing"] += 1; print("MISSING from the target:", name, flush=True); return
    off, nb, ty, shape = lay[name]
    if ty == gguf.GGMLQuantizationType.Q4_0:
        if raw_dtype == gguf.GGMLQuantizationType.BF16:
            f32 = (np.ascontiguousarray(tensor).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            f32 = np.ascontiguousarray(tensor, dtype=np.float32)
        dst = quant(f32, shape[0], nb)
        stats["q4"] += 1; stats["bytes"] += nb
        if a.check:
            cur = np.frombuffer(os.pread(fd, nb, off), np.uint8)
            same = np.array_equal(cur, dst)
            stats["ok" if same else "diff"] += 1
            if not same:
                print(f"DIFF {name}: {np.count_nonzero(cur != dst)}/{nb} bytes", flush=True)
        else:
            os.pwrite(fd, dst.tobytes(), off)
    else:
        raw = np.ascontiguousarray(tensor).tobytes()
        cur = os.pread(fd, nb, off)
        k = "other_same" if raw == cur else "other_diff"
        stats[k] += 1
        if k == "other_diff":
            print(f"other type {name} ({ty.name}): bytes differ from the skeleton (left as they are)", flush=True)
    if stats["q4"] % 20 == 0:
        print(f"{time.time() - t0:7.0f} s  {stats}", flush=True)


gguf.GGUFWriter.add_tensor = hook
for m in ("write_header_to_file", "write_kv_data_to_file", "write_tensors_to_file", "write_ti_data_to_file", "close", "open_output_file"):
    if hasattr(gguf.GGUFWriter, m):
        setattr(gguf.GGUFWriter, m, lambda self, *x, **k: None)
import convert_hf_to_gguf as C
sys.argv = ["convert_hf_to_gguf.py", a.hf, "--outtype", "bf16", "--no-lazy", "--outfile", "/dev/null"]
try:
    C.main()
except SystemExit:
    pass
missing = [n for n, v in lay.items() if v[2] == gguf.GGMLQuantizationType.Q4_0 and n not in seen and not (a.only and not re.search(a.only, n))]
print(f"DONE {time.time() - t0:.0f} s {stats}; Q4_0 tensors of the target never produced: {len(missing)} {missing[:5]}", flush=True)
os.close(fd)
