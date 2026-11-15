#!/bin/bash
# Exécute chaque dossier EXEC_DIR/<bras>/<tâche> dans un conteneur podman jetable (Ryzen 7 5700G) :
# pas de réseau, pas de home, dossier de la tâche en lecture seule, tmpfs, 1 Go, 128 processus, utilisateur nobody.
# _executer.sh EXEC_DIR BRAS...
set -u
ED=$1; shift
echo "== $(date '+%F %T %Z') $(hostname) podman $(podman --version | awk '{print $3}')"
for b in "$@"; do for d in "$ED/$b"/*/; do
  i=$(basename "$d"); L="$ED/$b/$i.log"
  C=(podman run --rm --network none --memory 1g --memory-swap 1g --pids-limit 128 --cpus 2 --read-only
     --tmpfs /tmp:rw,size=64m --tmpfs /work:rw,exec,size=768m -v "$(realpath "$d")":/task:ro
     --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1)
  if [ -f "$d/_lanceur.py" ]; then
    timeout 180 "${C[@]}" docker.io/library/python:3.12-slim python -I /task/_lanceur.py > "$L" 2>&1
  else
    timeout 300 "${C[@]}" docker.io/library/rust:1.82-slim sh /task/_lanceur_rs.sh > "$L" 2>&1
  fi
  echo "$b $i code=$? $(grep -c '@@TEST@@.*@@ok' "$L") $(grep -o '@@RESULTAT@@.*' "$L" | head -c 200)"
done; done
echo "== FIN $(date '+%T')"
