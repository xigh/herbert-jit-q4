#!/bin/bash
# ik_llama.cpp Q4_0 pur, relancé seul (le premier passage attendait des ids que ik ne rend pas). Sous flock ~/bench-ryzen-9-9900x.lock.
set -u
D=~/reponses-1114; cd $D; PY=~/venvs/tf522-lfm3/bin/python; M=~/models/Qwen3.6-35B-A3B
IS=~/bench-engines/ik_llama.cpp/build/bin/llama-server; Q40=~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf; PORT=32475
echo "== début $(date -u +%H:%M:%S) UTC"; free -g
$IS -m $Q40 -t 12 -c 4096 -np 1 --host 127.0.0.1 --port $PORT > logs/srv-ik-q40pur-2.log 2>&1 & SP=$!
for i in $(seq 1 900); do curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
echo "== $(date -u +%H:%M:%S) ik prêt (pid $SP)"
$PY _srv_gen.py ik http://127.0.0.1:$PORT prompts.jsonl out/ik-q40pur.raw.jsonl 2>&1 | tee logs/gen-ik-q40pur-2.log
kill $SP; sleep 3; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
$PY _ik_ids.py $M out/ik-q40pur.raw.jsonl out/ik-q40pur.jsonl 2>&1 | tee logs/ik-ids.log
echo "== FIN $(date -u +%H:%M:%S)"
