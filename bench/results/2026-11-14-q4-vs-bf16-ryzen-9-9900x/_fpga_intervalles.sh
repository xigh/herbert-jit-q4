#!/bin/bash
# Intervalles des travaux FPGA en cours sur la même machine (~/another-project sur Ryzen 9 9900X, run_recorded.py) : début = mtime de result.json
# moins duration_seconds, fin = mtime ; travaux de plus de 5 s finis après 20:30 ; les « build-command » (enveloppes de
# plusieurs étapes) sont retirés pour ne pas compter deux fois. Les travaux en cours se lisent dans ps (lstart) et s'ajoutent à la main.
ssh ryzen-9-9900x 'cd ~/another-project; find . -maxdepth 3 -name result.json -newermt "2026-11-14 20:30" | while read f; do python3 -c "
import json,sys,os;r=json.load(open(sys.argv[1]));e=os.path.getmtime(sys.argv[1]);d=r.get(\"duration_seconds\",0)
if d>5: print(round(e-d),round(e),round(d),sys.argv[1][2:])" $f; done | sort -n' | grep -v build-command
