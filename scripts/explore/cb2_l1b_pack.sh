# CB2-L1b packing, sourced by cb2_l1b_pod.sh (and tested by tests/test_cb2_l1b_pack.py).  Every log a later phase
# needs travels INSIDE the run directory ($OUT/logs), so the final phase works on a fresh CPU pod that only received
# cb2_l1b_loop.tgz (Sol, S-013).  Arguments: W (workspace), CODE (checkout), OUT (run dir, relative to CODE).

l1b_pack_loop() {   # after PHASE=run: the run directory, with the loop log inside it
  local W=$1 CODE=$2 OUT=$3
  mkdir -p "$CODE/$OUT/logs"
  if [ -f "$W/l1b-run.log" ]; then cp "$W/l1b-run.log" "$CODE/$OUT/logs/run.log"; fi
  tar -C "$CODE" -czf "$W/cb2_l1b_loop.tgz" "$OUT"
  sha256sum "$W/cb2_l1b_loop.tgz"
}

l1b_restore_loop() {   # before PHASE=final: the run directory as the loop left it (already present on the loop's pod)
  local W=$1 CODE=$2 OUT=$3
  if [ -f "$CODE/$OUT/run.json" ]; then return 0; fi
  if [ ! -f "$W/cb2_l1b_loop.tgz" ]; then echo "STOP: no loop results ($W/cb2_l1b_loop.tgz)"; return 1; fi
  tar --no-same-owner -C "$CODE" -xzf "$W/cb2_l1b_loop.tgz" "$OUT"
  [ -f "$CODE/$OUT/run.json" ] || { echo "STOP: cb2_l1b_loop.tgz holds no $OUT/run.json"; return 1; }
}

l1b_pack_final() {   # after PHASE=final: the run directory (all logs inside) + the l1b evidence files
  local W=$1 CODE=$2 OUT=$3
  mkdir -p "$CODE/$OUT/logs"
  if [ -f "$W/l1b-final.log" ]; then cp "$W/l1b-final.log" "$CODE/$OUT/logs/final.log"; fi
  local ev
  ev=$(cd "$CODE" && ls -d evidence/explore/cb2/l1b-* 2>/dev/null || true)
  # shellcheck disable=SC2086
  tar -C "$CODE" -czf "$W/cb2_l1b_results.tgz" "$OUT" $ev
  sha256sum "$W/cb2_l1b_results.tgz"
}
