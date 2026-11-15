#!/bin/bash
# Référence transformers bf16 (même environnement et même script que 2026-11-14-reponses-cote-a-cote), 20 invites, sur Ryzen 9 9900X,
# sous flock ~/bench-ryzen-9-9900x.lock. Attend ~75 Go disponibles (le serveur du port 8620 reste en place).
set -u
D=~/long-1114; cd $D; mkdir -p out logs
M=~/models/Qwen3.6-35B-A3B; PY=~/venvs/tf522-lfm3/bin/python
echo "== début $(date -u +%H:%M:%S) UTC, $(hostname)"; uptime
for i in $(seq 1 720); do A=$(free -g | awk '/Mem:/{print $7}'); [ $A -ge 75 ] && break; echo "attente RAM : $A Go disponibles ($(date -u +%H:%M:%S))"; sleep 60; done
free -g
$PY _ref_gen.py $M prompts.jsonl out/ref-tf-bf16.jsonl --threads 12 2>&1 | tee logs/ref-tf-bf16.log
echo "== FIN $(date -u +%H:%M:%S)"
