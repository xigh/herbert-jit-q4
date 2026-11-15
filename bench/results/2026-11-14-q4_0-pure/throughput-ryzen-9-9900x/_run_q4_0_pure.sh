#!/bin/bash
# Banc Ryzen 9 9900X (Ryzen 9 9900X, 12 fils), herbert avec ignore_eos (d24603d), sans suffixe d invite, Qwen3.6-35B-A3B, même fenêtre : herbert (--q4 --q4-grid8, --q4) contre llama.cpp 5676dfbe0 et
# ik_llama.cpp 7aa7e014 sur le GGUF Q4_0 pur (ik avec et sans -rtr), et llama.cpp Q4_K_M en rappel.
# Un serveur à la fois ; lancer sous flock ~/bench-epyc-4344p.lock. WORDS en argument (ex. "380 1500").
set -u
D=~/banc-q40/debit; mkdir -p $D
H=~/herbert-ignore-eos/engines/herbert-jit-q4/target/release/herbert-jit-llama-serve
L=~/bench-engines/llama.cpp-5676dfbe0/build/bin
K=~/bench-engines/ik_llama.cpp/build/bin
M=~/models/Qwen3.6-35B-A3B
F=~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf
KM=~/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
T=12; PORT=31000; ARMS=${ARMS:-all}
# 14/11 : un port neuf par serveur (ik_llama.cpp refuse de se lier au port du serveur précédent encore en TIME_WAIT).
export WORDS="$1"; TAG=$2; export SUFFIX=''
wait_up() { for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
run() { local label=$1; shift
  [ "$ARMS" = all ] || [[ " $ARMS " == *" $label "* ]] || return 0
  PORT=$((PORT + 1))
  echo "=== $label $TAG $(date -u +%T) : $*"
  set -- "${@//PORTX/$PORT}"; "$@" > $D/server-$label-$TAG.log 2>&1 &
  local pid=$!
  if wait_up $pid; then python3 ~/banc-q40/throughput.py http://127.0.0.1:$PORT $label $D/res-$label-qwen36-$TAG.json 2>&1 | tee -a $D/banc.log
  else echo "ECHEC démarrage $label"; tail -n 20 $D/server-$label-$TAG.log; fi
  kill $pid; wait $pid 2>/dev/null; sleep 5; }
echo "BANC $TAG WORDS=$WORDS $(date -u +%T) ; $(uptime)"
run herbert-q4g8 $H $M --threads $T --q4 --q4-grid8 --host 127.0.0.1 --port PORTX --ctx 8192
run llama-q40pur $L/llama-server -m $F -t $T -c 8192 -np 1 --host 127.0.0.1 --port PORTX
run ik-q40pur $K/llama-server -m $F -t $T -c 8192 -np 1 --host 127.0.0.1 --port PORTX
run ikrtr-q40pur $K/llama-server -m $F -rtr -t $T -c 8192 -np 1 --host 127.0.0.1 --port PORTX
run herbert-q4 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
run llama-q4km $L/llama-server -m $KM -t $T -c 8192 -np 1 --host 127.0.0.1 --port PORTX
echo "BANC $TAG FINI $(date -u +%T)"
