#!/bin/bash
# The arms of one model against its reference (step 4 of bench/quality): each engine scores the same
# tokens, then cmp.py (full reference ref/NAME-CORPUS.kld) or cmp2.py (its sketch ref/NAME-CORPUS.npz)
# compares it with the reference.
#
# arms.sh NAME HF_DIR CORPUS CHUNKS ARM...
#   NAME     the model's short name, as in ref/NAME-CORPUS.ids (ids.py) and the reference
#   HF_DIR   the model folder (safetensors) that hkld loads
#   CORPUS   wiki (corpus/wiki.test.raw) or fr (corpus/fr-miserables-t1.txt)
#   CHUNKS   chunks scored (48 for wiki and 24 for fr in the campaign)
#   ARM      herbert:LABEL:FLAGS   hkld kld, FLAGS comma-separated (herbert:q4:--q4)
#            llama:LABEL:GGUF      llama-perplexity of $LLAMA on that GGUF (-c 256 -b 256)
#            ik:LABEL:GGUF         the same with ik_llama.cpp's ($IK)
#            ikrtr:LABEL:GGUF      ik_llama.cpp with -rtr (weights repacked at load)
#
# env: WORK (holds corpus/ and ref/, default .), HKLD (hkld binary, default the workspace build),
#      PY (the comparator's python, numpy), TH (threads, default 12), LLAMA (llama.cpp's bin directory,
#      required), IK (ik_llama.cpp's, default $LLAMA), KEEP=1 keeps each arm's .kld.
# Writes out/NAME-CORPUS-LABEL.json (and .npz) and logs/NAME-CORPUS-LABEL.log; llama-perplexity tokenizes
# the corpus itself: tokcheck.py checks that it gets the ids of ref/NAME-CORPUS.ids.
S=$(cd "$(dirname "$0")" && pwd); cd "${WORK:-.}"; mkdir -p out logs
PY=${PY:-python3}; TH=${TH:-12}
LLAMA=${LLAMA:?llama.cpp bin directory}; IK=${IK:-$LLAMA}; HKLD=${HKLD:-$S/../../../target/release/hkld}
N=$1; M=$2; K=$3; CH=$4; shift 4
if [ $K = wiki ]; then T=corpus/wiki.test.raw; else T=corpus/fr-miserables-t1.txt; fi
if [ -f ref/$N-$K.kld ]; then CMP="$S/cmp.py ref/$N-$K.kld"; else CMP="$S/cmp2.py ref/$N-$K.npz"; fi
# Before any arm runs: without a reference, hours of scoring would be thrown away.
[ -f "${CMP#* }" ] || { echo "no reference ${CMP#* }: ref_tf.py (and sketch.py) first" >&2; exit 1; }
for A in "$@"; do
  E=${A%%:*}; R=${A#*:}; L=${R%%:*}; X=${R#*:}
  O=out/$N-$K-$L; echo "== $(date -u +%H:%M:%S) $N $K $L ($E $X)"
  case $E in
    herbert) $HKLD kld $M ref/$N-$K.ids $O.kld --threads $TH --chunks $CH ${X//,/ } > logs/$N-$K-$L.log 2>&1; TP="";;
    llama|ik|ikrtr) B=$LLAMA; XA=""; [ $E != llama ] && B=$IK; [ $E = ikrtr ] && XA=-rtr
       $B/llama-perplexity -m $X -f $T --kl-divergence-base $O.kld -c 256 -b 256 -t $TH --chunks $CH $XA > logs/$N-$K-$L.log 2>&1
       TP=$(grep -o "Final estimate: PPL = [0-9.]*" logs/$N-$K-$L.log | grep -o "[0-9.]*$"); TP=${TP:+--test-ppl $TP};;
  esac
  tail -1 logs/$N-$K-$L.log | cut -c1-200
  $PY $CMP $O.kld $O.json $TP | cut -c1-600
  [ -n "$KEEP" ] || rm -f $O.kld
done
