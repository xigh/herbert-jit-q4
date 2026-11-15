#!/bin/bash
# Q4 contre bf16, même code (68c3a96, ~/publication-jit-q4), même fenêtre, EPYC 4344P (EPYC 4344P, 8 fils), 14/11/2026.
# Par modèle : fichiers relus à chaud (cat > /dev/null) puis bras alternés bf16, Q4, Q4, bf16 (ABBA), un serveur à la fois, port neuf.
# Par serveur : temps lancement → /health (sondage 0,1 s), /proc/PID/status au repos, remise à zéro de VmHWM (clear_refs 5),
# banc (Qwen : ~/q40pur/throughput.py avec suffixe, comme le bloc A des rejeux ; Gemma : ~/banc-1113/throughput.py, comme le bloc C),
# échantillonnage de /proc/PID/status toutes les 0,5 s pendant le banc, /proc/PID/status à la fin.
# Toute la chaîne sous flock ~/bench-epyc-4344p.lock (lancée par l'appelant).
set -u
R=~/q4-contre-bf16
H=~/publication-jit-q4/engines/herbert-jit-q4/target/release/herbert-jit-llama-serve
T=8; PORT=33100
st() { grep -E '^(VmRSS|VmHWM|RssAnon|RssFile|RssShmem|VmSwap)' /proc/$1/status | tr -s ' \t' ' ' | tr '\n' ';'; }
run() { local label=$1 py=$2; shift 2
  PORT=$((PORT + 1))
  echo "=== $label $(date -u +%T) : $*"; uptime; free -m | sed -n 2p
  set -- "${@//PORTX/$PORT}"
  local t0=$(date +%s.%N)
  "$@" > $D/server-$label.log 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 6000); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 0.1; done
  local t1=$(date +%s.%N)
  if [ $ok = 1 ]; then
    echo "LOAD $label $(echo "$t1 - $t0" | bc) s"
    sleep 2; echo "REST $label $(st $pid)"
    echo 5 > /proc/$pid/clear_refs && echo "HWM-RESET $label $(st $pid)"
    ( while kill -0 $pid 2>/dev/null; do echo "$(date +%s.%N) $(st $pid)"; sleep 0.5; done ) > $D/rss-$label.log 2>&1 &
    local mon=$!
    python3 $py http://127.0.0.1:$PORT $label $D/res-$label.json 2>&1
    echo "END $label $(st $pid)"
    kill $mon
  else echo "ECHEC démarrage $label"; tail -n 30 $D/server-$label.log; fi
  kill $pid; wait $pid 2>/dev/null; sleep 5; }
warm() { local t0=$(date +%s.%N); cat "$@" > /dev/null; echo "WARM $(echo "$(date +%s.%N) - $t0" | bc) s : $*"; }
echo "Q4 CONTRE BF16 EPYC 4344P $(date -u +%T) ; $(uptime) ; code $(cat ~/publication-jit-q4/ARCHIVE-COMMIT.txt) ; $(sha256sum $H)"
# --- Qwen3.5-2B (Qwen3.5-4B absent de EPYC 4344P), protocole du bloc A : suffixe, 380 et 4240 mots
M=~/models/qwen/Qwen3.5-2B; D=$R/qwen35-2b; mkdir -p $D
export WORDS="380 4240"; export SUFFIX=$'\n1. Le port de Marseille'; P=~/q40pur/throughput.py
warm $M/*.safetensors
run bf16-qwen35-2b-r1 $P $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
run q4-qwen35-2b-r1 $P $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
run q4-qwen35-2b-r2 $P $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
run bf16-qwen35-2b-r2 $P $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
unset SUFFIX
# --- Gemma 4 E4B (safetensors QAT unquantized, bf16), protocole du bloc C : 78 (froid), 380, 4240 mots
M=~/models/gemma/gemma-4-E4B-it-qat-q4_0-unquantized; D=$R/gemmae4b; mkdir -p $D
export WORDS="78 380 4240"; P=~/banc-1113/throughput.py
warm $M/*.safetensors
run bf16-gemmae4b-r1 $P $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
run q4-gemmae4b-r1 $P $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
run q4-gemmae4b-r2 $P $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
run bf16-gemmae4b-r2 $P $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
# --- Qwen3.6-35B-A3B, Q4 seul (le bf16 ne tient pas), pour la mémoire ; 4240 mots avec suffixe
M=~/models/qwen/Qwen3.6-35B-A3B; D=$R/qwen36-35b; mkdir -p $D
export WORDS="4240"; export SUFFIX=$'\n1. Le port de Marseille'; P=~/q40pur/throughput.py
run q4-qwen36-35b $P $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
echo "Q4 CONTRE BF16 EPYC 4344P FINI $(date -u +%T)"
