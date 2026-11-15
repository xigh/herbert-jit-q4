#!/bin/bash
# Bras Q4 (serveurs, l'un après l'autre) puis référence transformers bf16. Sous flock ~/bench-ryzen-9-9900x.lock.
# _chaine_ryzen-9-9900x.sh   (dans ~/reponses-1114)
set -u
D=~/reponses-1114; cd $D; mkdir -p out logs
M=~/models/Qwen3.6-35B-A3B; PY=~/venvs/tf522-lfm3/bin/python; T=12
H=~/publication-jit-q4-rejeu/engines/herbert-jit-q4/target/release/herbert-jit-llama-serve
LS=~/bench-engines/llama.cpp-5676dfbe0/build/bin/llama-server
IS=~/bench-engines/ik_llama.cpp/build/bin/llama-server
Q40=~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf; QKM=~/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
echo "== début $(date -u +%H:%M:%S) UTC, $(hostname), $(grep -m1 'model name' /proc/cpuinfo), avx512f: $(grep -c avx512f /proc/cpuinfo)"
free -g
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
srv herbert-q40 herbert 32471 $H $M --q4 --threads $T --host 127.0.0.1 --ctx 4096 --log-dir $D/logs/herbert-req
srv llama-q40pur llama 32472 $LS -m $Q40 -t $T -c 4096 -np 1 --host 127.0.0.1
srv llama-q4km llama 32473 $LS -m $QKM -t $T -c 4096 -np 1 --host 127.0.0.1
srv ik-q40pur llama 32474 $IS -m $Q40 -t $T -c 4096 -np 1 --host 127.0.0.1
# Référence : attendre ~75 Go disponibles (le serveur du port 8620 reste en place).
for i in $(seq 1 720); do A=$(free -g | awk '/Mem:/{print $7}'); [ $A -ge 75 ] && break; echo "attente RAM : $A Go disponibles ($(date -u +%H:%M:%S))"; sleep 60; done
free -g
$PY _ref_gen.py $M prompts.jsonl out/ref-tf-bf16.jsonl --threads $T 2>&1 | tee logs/ref-tf-bf16.log
echo "== FIN $(date -u +%H:%M:%S)"
