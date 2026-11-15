#!/bin/bash
# The six arms of the pure-Q4_0 throughput comparison (Qwen3.6-35B-A3B), one server at a time, a fresh port per server:
# herbert --q4 (grid of Q4_0) and --q4 --q4-grid7, llama.cpp and ik_llama.cpp (with and without -rtr) on the pure Q4_0
# GGUF, llama.cpp on Q4_K_M as a reminder. The protocol of results/2026-11-14-*: --ctx 8192, 3 repetitions of 128 tokens.
#
#   HERBERT=target/release/herbert-jit-llama-serve MODEL=path/to/Qwen3.6-35B-A3B \
#   LLAMA=path/to/llama.cpp/build/bin IK=path/to/ik_llama.cpp/build/bin \
#   GGUF=Qwen3.6-35B-A3B-Q4_0-pure.gguf GGUF_Q4KM=Qwen3.6-35B-A3B-Q4_K_M.gguf \
#   THREADS=12 WORDS="78 380 4240" SUFFIX='' OUT=out bench/throughput/run_q4_0_pure.sh
#
# ARMS="herbert-g8 llama-q40pur" runs only those arms. The EPYC runs of results/2026-11-14-q4_0-pure/throughput-* used
# SUFFIX=$'\n1. Le port de Marseille' and 8 threads; the Ryzen runs SUFFIX='' and 12 threads. Run nothing else meanwhile.
set -u
here=$(cd "$(dirname "$0")" && pwd)
: "${HERBERT:?}" "${MODEL:?}" "${THREADS:?}" "${WORDS:?}"
OUT=${OUT:-out}; ARMS=${ARMS:-all}; PORT=${PORT:-32000}; export WORDS SUFFIX=${SUFFIX:-}
mkdir -p "$OUT"
wait_up() { for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
run() { local label=$1; shift
  [ "$ARMS" = all ] || [[ " $ARMS " == *" $label "* ]] || return 0
  PORT=$((PORT + 1))
  echo "=== $label $(date -u +%T) : $*"; uptime
  set -- "${@//PORTX/$PORT}"; "$@" > "$OUT/server-$label.log" 2>&1 &
  local pid=$!
  if wait_up $pid; then python3 "$here/throughput.py" http://127.0.0.1:$PORT $label "$OUT/res-$label.json" 2>&1 | tee -a "$OUT/bench.log"
  else echo "FAILED to start $label"; tail -n 20 "$OUT/server-$label.log"; fi
  kill $pid; wait $pid 2>/dev/null; sleep 5; }
run herbert-g8 "$HERBERT" "$MODEL" --threads $THREADS --q4 --host 127.0.0.1 --port PORTX --ctx 8192
[ -n "${LLAMA:-}" ] && run llama-q40pur "$LLAMA/llama-server" -m "$GGUF" -t $THREADS -c 8192 -np 1 --host 127.0.0.1 --port PORTX
[ -n "${IK:-}" ] && run ik-q40pur "$IK/llama-server" -m "$GGUF" -t $THREADS -c 8192 -np 1 --host 127.0.0.1 --port PORTX
[ -n "${IK:-}" ] && run ikrtr-q40pur "$IK/llama-server" -m "$GGUF" -rtr -t $THREADS -c 8192 -np 1 --host 127.0.0.1 --port PORTX
run herbert-g7 "$HERBERT" "$MODEL" --threads $THREADS --q4 --q4-grid7 --host 127.0.0.1 --port PORTX --ctx 8192
[ -n "${LLAMA:-}" ] && [ -n "${GGUF_Q4KM:-}" ] && run llama-q4km "$LLAMA/llama-server" -m "$GGUF_Q4KM" -t $THREADS -c 8192 -np 1 --host 127.0.0.1 --port PORTX
echo "done $(date -u +%T)"
