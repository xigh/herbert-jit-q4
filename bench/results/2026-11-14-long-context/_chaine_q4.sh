#!/bin/bash
# Bras Q4 (serveurs l'un après l'autre, contexte 8192, 150 jetons), sur Ryzen 9 9900X, sous flock ~/bench-ryzen-9-9900x.lock.
# Mêmes binaires, modèles et fils que 2026-11-14-reponses-cote-a-cote/_chaine_ryzen-9-9900x.sh ; ports neufs ; ik en kind « ik ».
# _chaine_q4.sh   (dans ~/long-1114)
set -u
D=~/long-1114; cd $D; mkdir -p out logs
M=~/models/Qwen3.6-35B-A3B; PY=~/venvs/tf522-lfm3/bin/python; T=12; C=8192
H=~/publication-jit-q4-rejeu/engines/herbert-jit-q4/target/release/herbert-jit-llama-serve
LS=~/bench-engines/llama.cpp-5676dfbe0/build/bin/llama-server
IS=~/bench-engines/ik_llama.cpp/build/bin/llama-server
Q40=~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf; QKM=~/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
echo "== début $(date -u +%H:%M:%S) UTC, $(hostname), $(grep -m1 'model name' /proc/cpuinfo), avx512f: $(grep -c avx512f /proc/cpuinfo)"
sha256sum $H $LS $IS
free -g; uptime
srv() {  # LABEL KIND PORT cmd...
  local L=$1 K=$2 PORT=$3; shift 3
  "$@" --port $PORT > logs/srv-$L.log 2>&1 & local SP=$!
  for i in $(seq 1 900); do curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  echo "== $(date -u +%H:%M:%S) $L prêt (pid $SP, port $PORT)"
  $PY _srv_gen.py $K http://127.0.0.1:$PORT prompts.jsonl out/$L.jsonl ${LOGD:-} 2>&1 | tee logs/gen-$L.log
  kill $SP; sleep 3; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "== $(date -u +%H:%M:%S) $L fini"
}
mkdir -p logs/herbert-req; LOGD=$D/logs/herbert-req \
srv herbert-q40 herbert 32481 $H $M --q4 --threads $T --host 127.0.0.1 --ctx $C --log-dir $D/logs/herbert-req
LOGD= srv llama-q40pur llama 32482 $LS -m $Q40 -t $T -c $C -np 1 --host 127.0.0.1
LOGD= srv llama-q4km llama 32483 $LS -m $QKM -t $T -c $C -np 1 --host 127.0.0.1
LOGD= srv ik-q40pur ik 32484 $IS -m $Q40 -t $T -c $C -np 1 --host 127.0.0.1
mv out/ik-q40pur.jsonl out/ik-q40pur.raw.jsonl
$PY _ik_ids.py $M out/ik-q40pur.raw.jsonl out/ik-q40pur.jsonl 2>&1 | tee logs/ik-ids.log
echo "== FIN $(date -u +%H:%M:%S)"
