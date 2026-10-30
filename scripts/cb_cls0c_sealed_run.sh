#!/usr/bin/env bash
# Run on David's PC (WSL). CLS-0C SEALED TEST: the ONE evaluation (lab/CLS0C_SEALED_PROTOCOL.md, results/cls0c_sealed/VALIDITY_AMENDMENT.md).
# Creates an A40 pod on volume 29qhoqghqf (CA-MTL-1) and uses a SEPARATE checkout pinned to the sealed commit; earlier checkouts
# and their results on the volume are not touched. Then:
#   preflight - pinned HEAD; harness files byte-identical to the 2b240fc freeze; sealed.json matches SEALED.sha256;
#               CLS-0B inputs match FILES.sha256; models present; no earlier RUN_STARTED marker
#   stage 1   - bridge-write seed 101 (CLS-0B storage config). Resumable: a relaunch continues from the bank on the volume
#   stage 2   - run (B1_K1 primary, B0 descriptive, natural-question challenge, adapter bridge). Runs ONCE: the harness writes
#               RUN_STARTED and refuses any second run
# Finally it downloads and verifies the bundle to ~/cb_results/cls0c_sealed and removes the pod.
# Exit codes: 0 complete; 1 anything else (preflight STOP incl. an existing RUN_STARTED, operational failure; pod stopped/removed,
# files stay on the volume). Stage timeouts: bridge 4 h, run 4 h; launcher cap 8.5 h (above both, so the cap never cuts the run).
#   bash cb_cls0c_sealed_run.sh [gpu="NVIDIA A40"]
GPU=${1:-NVIDIA A40}; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
PIN=0d8ac6450c08558dd145498f2e024f2aab464a9b                   # sealed commit (sealed.json b993ae9c..., amendment, cohorts)
FREEZE=2b240fc7e91592742dc24c2e094c2fcfdb1c0b1c                # harness freeze, made before any sealed item existed
D=/workspace/Continuous_brain_c0csealed; OUT=$HOME/cb_results/cls0c_sealed; DL=/mnt/c/Users/davei/Downloads
CAP=$((17 * 1800)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cls0c-sealed --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 20m > ~/cb_c0csealed_pod.json \
  || { echo "STOP: pod create failed"; cat ~/cb_c0csealed_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_c0csealed_pod.json"))["id"])') || exit 1
DONE=0                                                       # any unsuccessful exit after this point stops the pod
trap 'if [ "$DONE" != 1 ] && [ -n "$POD" ]; then echo "== cleanup: stopping pod $POD (files stay on the volume)"; runpodctl pod stop $POD; fi' EXIT
echo "== pod $POD ($GPU, $DC)"
addr() { eval "$(runpodctl ssh info $POD 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"; }
addr; S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
$S "VOL=$VOL D=$D PIN=$PIN FREEZE=$FREEZE bash -s" <<'REMOTE' || { echo "STOP: preflight/launch failed - removing pod"; runpodctl pod remove $POD && POD=""; exit 1; }
set -Eeuo pipefail
findmnt -no SOURCE /workspace | grep -q networkvolumes/$VOL
if pgrep -f '[s]cripts/cls0' >/dev/null; then echo 'STOP: a CLS process is running in this pod'; exit 1; fi
[ -d $D ] || git clone -q --depth 200 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D
if ! git diff --quiet || ! git diff --cached --quiet; then echo 'STOP: tracked files modified in the sealed checkout'; git status --short; exit 1; fi
git fetch -q --depth 200 origin being/v3-bind
git cat-file -e $PIN^{commit}; git cat-file -e $FREEZE^{commit}
git checkout -q --detach $PIN
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo 'STOP: HEAD is not the pinned sealed commit'; exit 1; }
echo "sealed code at: $(git log --oneline -1)"
git diff --quiet $FREEZE HEAD -- scripts/cls0c_sealed.py scripts/cls0c_b.py scripts/cls0b.py scripts/bind_i1.py being/ \
  || { echo 'STOP: harness differs from the 2b240fc freeze'; exit 1; }
(cd results/cls0c_sealed && sha256sum -c --quiet SEALED.sha256) || { echo 'STOP: sealed.json does not match SEALED.sha256'; exit 1; }
(cd results/cls0b_final && sha256sum -c --quiet FILES.sha256) || { echo 'STOP: CLS-0B inputs do not match FILES.sha256'; exit 1; }
[ ! -e results/cls0c_sealed_run/RUN_STARTED ] || { echo 'STOP: RUN_STARTED exists - the sealed run was already started once'; exit 1; }
ln -sfn /workspace/Continuous_brain/.venv .venv
for m in Qwen3-8B Qwen3-Embedding-0.6B; do [ -f /workspace/hf/$m/config.json ] || { echo "STOP: model $m missing"; exit 1; }; done
echo "PREFLIGHT OK: sealed $(cut -c1-16 results/cls0c_sealed/SEALED.sha256), harness == freeze, CLS-0B inputs verified"
rm -f /workspace/results_cls0c_sealed.tar.gz /workspace/results_cls0c_sealed.tar.gz.sha256
cat > /workspace/cls0c_sealed_job.sh <<'JOB'
cd /workspace/Continuous_brain_c0csealed
R=results/cls0c_sealed_run; M=/workspace/hf/Qwen3-8B; X=/workspace/hf/Qwen3-Embedding-0.6B
echo "== stage 1: bridge-write seed 101 $(date -u +%FT%TZ)"
timeout 14400 .venv/bin/python scripts/cls0c_sealed.py bridge-write --seed 101 --model $M --out $R/bridge; e1=$?
if [ $e1 -eq 0 ] && [ -f $R/bridge/bridge_seed101_recon.json ]; then
  echo "== stage 2: sealed run $(date -u +%FT%TZ)"
  timeout 14400 .venv/bin/python scripts/cls0c_sealed.py run results/cls0c_sealed --model $M --external $X --out $R \
      --bridge $R/bridge/bridge_seed101_recon.json; e=$?
else
  echo "bridge-write did not finish (exit $e1) - sealed run NOT started; relaunching resumes the bridge"; e=$((e1 ? e1 : 1))
fi
echo "== end $(date -u +%FT%TZ)"
tar czf /workspace/results_cls0c_sealed.tar.gz $R results/cls0c_sealed
sha256sum /workspace/results_cls0c_sealed.tar.gz | sed 's|/workspace/||' > /workspace/results_cls0c_sealed.tar.gz.sha256 && sync
echo "BUNDLE $(cat /workspace/results_cls0c_sealed.tar.gz.sha256)"
echo "CLS-0C-SEALED JOB DONE exit=$e"
JOB
nohup bash /workspace/cls0c_sealed_job.sh > /workspace/cls0c_sealed.log 2>&1 < /dev/null &
sleep 5; echo "SEALED JOB LAUNCHED"
REMOTE
echo "== waiting for the sealed job (checks every 2 min, cap 8.5 h)"
while :; do
  sleep 120; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CLS-0C-SEALED JOB DONE' /workspace/cls0c_sealed.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 /workspace/cls0c_sealed.log | cut -c1-100" 2>/dev/null)"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 8.5-hour cap reached - stopping pod (files stay on the volume)"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "grep -v 'Loading weights\|generation flags\|attention mask\|bridge write' /workspace/cls0c_sealed.log"
mkdir -p $OUT
$S "cd /workspace && tar cf - results_cls0c_sealed.tar.gz results_cls0c_sealed.tar.gz.sha256 cls0c_sealed.log" | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c results_cls0c_sealed.tar.gz.sha256) \
  && cp $OUT/results_cls0c_sealed.tar.gz $DL/ \
  && echo "== downloaded + verified: $OUT (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
E=$(grep -o 'CLS-0C-SEALED JOB DONE exit=[0-9]*' $OUT/cls0c_sealed.log | tail -n 1 | cut -d= -f2)
echo "== CLS-0C-SEALED exit: ${E:-UNKNOWN}"
[ "$E" = 0 ] && exit 0 || exit 1
