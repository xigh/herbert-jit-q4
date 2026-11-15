#!/bin/bash
# run.sh PYTHON [OUT_DIR]: the reference render (PYTHON must have transformers and jinja2), ours, the comparison.
# From herbert-chat/conformance; OUT_DIR defaults to ./out.
set -u; cd "$(dirname "$0")"; PY=$1; OUT=${2:-out}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
"$PY" -I reference.py cases.json templates "$OUT/reference.json" || exit 2
(cd .. && cargo run --release --quiet --example conformance -- conformance/cases.json conformance/templates "$OUT/ours.json") || exit 2
"$PY" -I compare.py "$OUT/reference.json" "$OUT/ours.json" "$OUT/report.txt"
