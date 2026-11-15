#!/bin/bash
# The pure Q4_0 GGUF of Qwen3.6-35B-A3B that llama.cpp and ik_llama.cpp were measured on: llama.cpp 5676dfbe0's
# convert_hf_to_gguf.py to bf16, then `llama-quantize --pure ... Q4_0` (no imatrix), then its sha256, to compare with
# sha256-qwen36-q40pur.txt (a convert of another day adds the MTP layer: tensor_sha.py and layout.py compare and
# realign the 733 common tensors, results/2026-11-14-q4_0-pure/tensor-sha-*.txt).
#   LLAMA_CPP=path/to/llama.cpp PY=python3 make_q4_0_pure.sh path/to/Qwen3.6-35B-A3B OUT_DIR
set -eu
L=${LLAMA_CPP:?}; PY=${PY:-python3}; M=$1; O=$2; mkdir -p "$O"
B=$O/Qwen3.6-35B-A3B-BF16.gguf; F=$O/Qwen3.6-35B-A3B-Q4_0-pure.gguf
$PY "$L/convert_hf_to_gguf.py" "$M" --outtype bf16 --outfile "$B"
"$L/build/bin/llama-quantize" --pure "$B" "$F" Q4_0 "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
sha256sum "$F" 2>/dev/null || shasum -a 256 "$F"
