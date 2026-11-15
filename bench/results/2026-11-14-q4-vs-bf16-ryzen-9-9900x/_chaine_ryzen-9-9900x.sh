#!/bin/bash
# Chaîne des segments, un verrou par modèle : avant chaque prise, attendre qu'aucun flock en attente ne vienne de l'emballage
# (cwd ~/emballage-archive) ni d'une compilation/test cargo, pour les laisser passer ; puis flock ~/bench-ryzen-9-9900x.lock pour un seul modèle.
set -u
cd ~/q4-contre-bf16-ryzen-9-9900x
gate() { while :; do local busy=""
  for p in $(lslocks -n -o PID,PATH | awk '/bench-ryzen-9-9900x.lock/{print $1}'); do
    [ "$p" = "$$" ] && continue
    local w=$(readlink /proc/$p/cwd 2>/dev/null) c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$w $c" in *emballage*|*cargo*) busy="$busy $p";; esac
  done
  [ -z "$busy" ] && return; echo "ATTENTE emballage/cargo :$busy $(date +%T)"; sleep 30; done; }
echo "CHAINE Ryzen 9 9900X $(date +%T) ; $(uptime)"
# Garde de repos aussi avant de prendre le verrou (ne pas le tenir pendant que le serveur 8620 travaille).
idle() { local p=$(ss -ltnpH 'sport = :8620' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2) a b l
  if [ -n "$p" ]; then a=$(awk '{print $14+$15}' /proc/$p/stat); sleep 5; b=$(awk '{print $14+$15}' /proc/$p/stat); else a=0; b=0; sleep 5; fi
  l=$(cut -d' ' -f1 /proc/loadavg); echo "REPOS? 8620 $((b - a)) ticks/5 s ; charge $l"
  [ $((b - a)) -lt 25 ] && awk -v l=$l 'BEGIN{exit !(l < 3)}'; }
for m in "$@"; do
  until idle; do echo "ATTENTE-REPOS $(date +%T)"; sleep 60; done
  gate
  echo "FILE $m $(date +%T) : en attente du verrou"
  flock ~/bench-ryzen-9-9900x.lock ./_q4_bf16_ryzen-9-9900x.sh $m
  echo "VERROU RENDU $m $(date +%T)"; sleep 20
done
echo "CHAINE Ryzen 9 9900X FINIE $(date +%T)"
