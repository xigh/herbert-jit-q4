#!/bin/sh
# Lanceur Rust dans le conteneur : copie le crate (lecture seule) dans un tmpfs, compile hors ligne, passe chaque test seul (10 s).
set -u
cp -r /task/crate /work/crate && cd /work/crate
export CARGO_HOME=/work/cargo CARGO_TARGET_DIR=/work/target
if timeout 120 cargo test --offline --no-run > /work/build.log 2>&1; then echo "@@COMPILE@@ok"; else echo "@@COMPILE@@non"; cat /work/build.log; exit 0; fi
cat /work/build.log
BIN=$(ls /work/target/debug/deps/sol-* | grep -v '\.d$' | head -n 1)
for t in $($BIN --list | sed -n 's/: test$//p'); do
  if timeout 10 $BIN --exact "$t" --test-threads=1 > /work/t.log 2>&1; then echo "@@TEST@@$t@@ok"; else echo "@@TEST@@$t@@échec($?)"; cat /work/t.log; fi
done
