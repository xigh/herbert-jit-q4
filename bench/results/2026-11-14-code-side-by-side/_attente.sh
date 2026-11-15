#!/bin/bash
# Attend que le serveur du port 8620 (PID 3221320) soit au repos (3 relevés d'une minute sous 30 % d'un cœur) ou parti,
# puis lance la chaîne sous ~/bench-ryzen-9-9900x.lock (flock attend s'il est pris). La référence f32 a été annulée
# (20 h 10) : plus d'attente de son chronométrage. ~/code-1114/GO force le départ.
set -u
cd ~/code-1114
cpu() { awk '{print $14+$15}' /proc/3221320/stat 2>/dev/null || echo X; }
echo "== attente $(date '+%F %T %Z')"
calme=0
while true; do
  a=$(cpu); sleep 60; b=$(cpu)
  if [ "$a" = X ] || [ "$b" = X ]; then echo "$(date '+%T') 8620 parti"; break; fi
  pct=$(( (b - a) * 100 / 6000 ))
  if [ $pct -lt 30 ]; then calme=$((calme+1)); else calme=0; fi
  echo "$(date '+%T') 8620 ${pct}% (repos $calme/3), verrou $(flock -n ~/bench-ryzen-9-9900x.lock true && echo libre || echo pris)"
  [ $calme -ge 3 ] && { echo "8620 au repos"; break; }
  [ -e GO ] && { echo "GO posé : $(cat GO)"; break; }
done
echo "== $(date '+%T') flock"
flock ~/bench-ryzen-9-9900x.lock ./_chaine_ryzen-9-9900x.sh
echo "== chaîne rendue $(date '+%T')"
