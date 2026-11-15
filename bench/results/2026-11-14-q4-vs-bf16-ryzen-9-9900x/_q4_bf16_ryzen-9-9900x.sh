#!/bin/bash
# Q4 contre bf16 sur Ryzen 9 9900X (Ryzen 9 9900X, 12 fils), 14/11/2026 : copie de _q4_bf16_epyc-4344p.sh (2026-11-14-q4-contre-bf16), adaptée.
# Un segment = un modèle (argument : 4b, 2b, e4b, 35b), lancé sous flock ~/bench-ryzen-9-9900x.lock par _chaine_ryzen-9-9900x.sh, pour rendre
# le verrou entre deux modèles. Mêmes mesures que EPYC 4344P : fichiers relus à chaud puis bf16, Q4, Q4, bf16 (ABBA), un serveur à
# la fois, port neuf ; temps lancement → /health ; /proc/PID/status au repos ; clear_refs 5 ; banc ; status toutes les 0,5 s ; status final.
# Différences avec EPYC 4344P : 12 fils ; client unique ~/banc-q40/throughput.py (md5 367da…, = ~/q40pur/throughput.py de EPYC 4344P ; sans SUFFIX,
# il envoie les mêmes invites que ~/banc-1113/throughput.py, absent de Ryzen 9 9900X) ; serveurs du banc lancés sous choom -n 1000 (un OOM
# les tue avant le serveur du port 8620) ; état du serveur 8620 (RSS, ticks CPU utime+stime) avant et après chaque bras ;
# 35B : garde MemAvailable avant le bf16.
set -u
R=~/q4-contre-bf16-ryzen-9-9900x
H=~/publication-jit-q4-rejeu/engines/herbert-jit-q4/target/release/herbert-jit-llama-serve
P=~/banc-q40/throughput.py
T=12
st() { grep -E '^(VmRSS|VmHWM|RssAnon|RssFile|RssShmem|VmSwap)' /proc/$1/status | tr -s ' \t' ' ' | tr '\n' ';'; }
p8620() { local p=$(ss -ltnpH 'sport = :8620' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -n "$p" ]; then echo "pid $p rss_kB $(awk '/VmRSS/{print $2}' /proc/$p/status) ticks $(awk '{print $14+$15}' /proc/$p/stat)"; else echo "absent"; fi; }
# Garde de repos (demandée à 19 h 45 : le serveur 8620 servait un panel à ~900 % de CPU) : avant chaque bras, le serveur 8620
# brûle moins de 5 % d'un cœur sur 5 s (ticks utime+stime) et la charge sur 1 min est sous 3 ; sinon attente de 60 s et nouvel essai.
# Garde révisée à 22 h 33 (consigne) : le port 8620 n'a pas de processus, et aucun processus autre que le travail FPGA
# en cours sur la même machine (nextpnr, yosys, run_recorded.py sous ~/another-project) ne dépasse 5 % de CPU sur 3 s. La présence du
# travail FPGA est notée à chaque bras (FPGA oui/non) ; la charge est dans la ligne uptime.
fpga() { local n=0; for p in $(pgrep -u user -f '[n]extpnr|[y]osys|[r]un_recorded'); do case "$(readlink /proc/$p/cwd 2>/dev/null)" in *another-project*) n=$((n+1));; esac; done; [ $n -gt 0 ] && echo oui || echo non; }
others() { top -b -n 2 -d 3 | awk '/^ *PID/{k++; next} k==2 && $9+0 > 5 {print $1}' | while read q; do
  local w="$(readlink /proc/$q/cwd 2>/dev/null) $(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null)"
  case "$w" in *another-project*|*nextpnr*|*yosys*) continue;; esac
  [ "$q" != "$$" ] && echo -n "$q:$(cat /proc/$q/comm 2>/dev/null) "; done; }
idle() { local p=$(ss -ltnpH 'sport = :8620' | grep -o 'pid=[0-9]*' | head -1) busy=$(others)
  echo "REPOS? 8620 ${p:-absent} ; autres >5 % : ${busy:-aucun} ; FPGA $(fpga) ; charge $(cut -d' ' -f1-3 /proc/loadavg)"
  [ -z "$p" ] && [ -z "$busy" ]; }
waitidle() { until idle; do echo "ATTENTE-REPOS $(date +%T)"; sleep 60; done; }
run() { local label=$1; shift
  waitidle
  PORT=$((PORT + 1))
  echo "=== $label $(date +%T) : $*"; uptime; free -m | sed -n 2p; grep MemAvailable /proc/meminfo
  echo "P8620-AVANT $label $(p8620)"; echo "FPGA-AVANT $label $(fpga)"
  ps -eo pid,pcpu,rss,etime,comm --sort=-pcpu | head -6
  set -- "${@//PORTX/$PORT}"
  local t0=$(date +%s.%N)
  choom -n 1000 -- "$@" > $D/server-$label.log 2>&1 &
  local pid=$! ok=0
  for i in $(seq 1 6000); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 0.1; done
  local t1=$(date +%s.%N)
  if [ $ok = 1 ]; then
    echo "LOAD $label $(echo "$t1 - $t0" | bc) s"
    sleep 2; echo "REST $label $(st $pid)"
    echo 5 > /proc/$pid/clear_refs && echo "HWM-RESET $label $(st $pid)"
    ( while kill -0 $pid 2>/dev/null; do echo "$(date +%s.%N) $(st $pid)"; sleep 0.5; done ) > $D/rss-$label.log 2>&1 &
    local mon=$!
    echo "P8620-BANC-DEBUT $label $(p8620)"
    python3 $P http://127.0.0.1:$PORT $label $D/res-$label.json 2>&1
    echo "P8620-BANC-FIN $label $(p8620)"; echo "FPGA-FIN $label $(fpga)"
    echo "END $label $(st $pid)"
    kill $mon
  else echo "ECHEC démarrage $label"; tail -n 30 $D/server-$label.log; fi
  kill $pid; wait $pid 2>/dev/null; sleep 5; }
warm() { local t0=$(date +%s.%N); cat "$@" > /dev/null; echo "WARM $(echo "$(date +%s.%N) - $t0" | bc) s : $*"; }
echo "SEGMENT $1 $(date +%T) ; $(uptime) ; code $(cat ~/publication-jit-q4-rejeu/ARCHIVE-COMMIT.txt) ; $(sha256sum $H) ; client $(md5sum $P)"
case $1 in
 4b|2b)
  if [ $1 = 4b ]; then M=~/models/Qwen3.5-4B; D=$R/qwen35-4b; PORT=34100; n=qwen35-4b; else M=~/models/Qwen3.5-2B; D=$R/qwen35-2b; PORT=34110; n=qwen35-2b; fi
  mkdir -p $D; export WORDS="380 4240"; export SUFFIX=$'\n1. Le port de Marseille'
  warm $M/*.safetensors
  run bf16-$n-r1 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
  run q4-$n-r1 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  run q4-$n-r2 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  run bf16-$n-r2 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192 ;;
 e4b)
  M=~/models/gemma-4-E4B-it-qat-q4_0-unquantized; D=$R/gemmae4b; PORT=34120; mkdir -p $D
  export WORDS="78 380 4240"; unset SUFFIX
  warm $M/*.safetensors
  run bf16-gemmae4b-r1 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
  run q4-gemmae4b-r1 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  run q4-gemmae4b-r2 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  run bf16-gemmae4b-r2 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192 ;;
 35b)
  # bf16 : ~72 Go de poids en mémoire anonyme ; garde : MemAvailable ≥ 95 Go avant chaque bras bf16, sinon le bras est sauté.
  M=~/models/Qwen3.6-35B-A3B; D=$R/qwen36-35b; PORT=34130; mkdir -p $D
  export WORDS="380 4240"; export SUFFIX=$'\n1. Le port de Marseille'
  g() { local a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ $a -ge 95 ]; then return 0; fi; echo "GARDE $1 : MemAvailable $a Gio < 95, bras sauté"; return 1; }
  warm $M/*.safetensors
  g bf16-qwen36-35b-r1 && run bf16-qwen36-35b-r1 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192
  run q4-qwen36-35b-r1 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  run q4-qwen36-35b-r2 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  g bf16-qwen36-35b-r2 && run bf16-qwen36-35b-r2 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192 ;;
 35b-suite)
  # reprise après l'arrêt de 22 h 32 : les deux bras restants, ports à la suite (34132 = q4-r1)
  M=~/models/Qwen3.6-35B-A3B; D=$R/qwen36-35b; PORT=34132; mkdir -p $D
  export WORDS="380 4240"; export SUFFIX=$'\n1. Le port de Marseille'
  g() { local a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ $a -ge 95 ]; then return 0; fi; echo "GARDE $1 : MemAvailable $a Gio < 95, bras sauté"; return 1; }
  run q4-qwen36-35b-r2 $H $M --threads $T --q4 --host 127.0.0.1 --port PORTX --ctx 8192
  g bf16-qwen36-35b-r2 && run bf16-qwen36-35b-r2 $H $M --threads $T --host 127.0.0.1 --port PORTX --ctx 8192 ;;
esac
echo "SEGMENT $1 FINI $(date +%T)"
