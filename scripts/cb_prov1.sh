#!/usr/bin/env bash
# Run on David's PC (WSL). PROV-1 (lab/PROV1_PROTOCOL.md). One launcher, three phases, each launched only after Claude's
# and Sol's GO for that phase:
#   bash cb_prov1.sh 1   calibration + timing: SUPPORT-1 reproducibility pilot (seed 24, P, sleep 1 vs its committed
#                        hash), PROV-1 RP/RC sleep-1 calibration (calcheck), measured timings -> phase-A/B cost projection
#   bash cb_prov1.sh A   zero-shot diagnostic: regenerate SUPPORT-1 seeds 20-23 with its frozen train stage, admit the
#                        44 P/U4 checkpoints only on exact sha256 equality with SUPPORT-1's committed train logs, evaluate
#                        them under the record conditions (twins), run both analyzers
#   bash cb_prov1.sh B   trained use: RP/RU/RC seeds 20-23 (+ RU/RC fork), every evaluation (twins, N, withheld), both analyzers
# Every phase first verifies phase 0 (committed preps archive == PHASE0.sha256, regenerated preps identical, Sol's
# validator PASS on this pod), the CODE_HASH, SUPPORT-1's frozen CODE_HASH, the model manifest and the GPU.
#   pod : NVIDIA RTX 6000 Ada Generation only (SUPPORT-1's card; phase A's hash equality depends on it), SECURE cloud
#   out : /mnt/e/cb_results/prov1/phase<P>/ (David's mounted E: drive; nothing is staged on C:). Results bundle + sha256
#         always; every checkpoint still on the pod (phase B; any mismatching evidence) streamed and sha256-verified. Evidence is preserved and verified before any
#         pod removal, else the pod is STOPPED and kept.
PHASE=${1:-}
case "$PHASE" in 1|A|B) ;; *) echo "usage: bash cb_prov1.sh 1|A|B"; exit 1;; esac
PIN=49201753bcfaafd4353080b8ed5f369e76b99fa8   # Sol GO on the phase-1 path at this commit (sol/prov1-phase1-go e06cefa)
REV=b968826d9c46dd6066d109eabc6255188de91218
GPU_NAME="NVIDIA RTX 6000 Ada Generation"
K=$HOME/.runpod/ssh/runpodctl-ssh-key; DL=/mnt/c/Users/davei/Downloads
OUT=/mnt/e/cb_results/prov1/phase$PHASE
D=/workspace/Continuous_brain_prov1; POD=""; S=""; DONE=0
KEEP_CKPT=1        # every .pt still on the pod is streamed to E: and verified. Phase 1/A delete their reproducible or
                   # hash-identical files after a MATCH, so only phase-B checkpoints and any mismatching evidence remain
case "$PHASE" in 1) CAP_H=3;; A) CAP_H=16;; B) CAP_H=24;; esac
[[ $PIN =~ ^[0-9a-f]{40}$ ]] || { echo "STOP: launcher PIN is not a resolved commit hash"; exit 1; }
# E: must be the mounted Windows drive itself (Sol): a plain directory at /mnt/e would live on WSL's disk, i.e. on C:
case "$(realpath -m "$OUT")" in /mnt/e/*) ;; *) echo "STOP: output path does not resolve under /mnt/e"; exit 1;; esac
[ "$(findmnt -n -o TARGET --target /mnt/e 2>/dev/null)" = /mnt/e ] && findmnt -n -o SOURCE /mnt/e 2>/dev/null | grep -qi '^E:' \
  || { echo "STOP: /mnt/e is not the mounted E: drive (is WD_BLACK connected?)"; exit 1; }
mkdir -p "$OUT" 2>/dev/null && touch "$OUT/.w" 2>/dev/null && rm -f "$OUT/.w" \
  || { echo "STOP: E: drive not writable at $OUT"; exit 1; }
[ "$(findmnt -n -o TARGET --target "$OUT" 2>/dev/null)" = /mnt/e ] || { echo "STOP: $OUT is not on the E: mount"; exit 1; }
FREE_GB=$(df -BG --output=avail /mnt/e | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 40 ] || { echo "STOP: less than 40 GB free on E: ($FREE_GB GB)"; exit 1; }
USD_PER_HR=${USD_PER_HR:-0.99}                            # RTX 6000 Ada secure rate observed in SUPPORT-1 phase 2

preserve() {  # results bundle (+ phase B checkpoints) streamed to E:, each verified against the pod's sha256; 0 only if verified
  $S "cd /workspace && tar czf prov1_results.tar.gz --exclude='*.pt' prov1 && sha256sum prov1_results.tar.gz > prov1_results.tar.gz.sha256 \
      && find prov1 -name '*.pt' | sort | xargs -r sha256sum > prov1_ckpt.sha256" \
  && $S "cat /workspace/prov1_results.tar.gz.sha256" > "$OUT/prov1_results.tar.gz.sha256" \
  && $S "cat /workspace/prov1_results.tar.gz" > "$OUT/prov1_results.tar.gz" \
  && (cd "$OUT" && sha256sum -c --quiet prov1_results.tar.gz.sha256) \
  && $S "cat /workspace/prov1_ckpt.sha256" > "$OUT/prov1_ckpt.sha256" \
  && if [ "$KEEP_CKPT" = 1 ]; then
       mkdir -p "$OUT/ckpt" \
       && $S "cd /workspace && cut -c67- prov1_ckpt.sha256 | tar cf - -T -" | tar xf - -C "$OUT/ckpt" --no-same-owner --no-same-permissions \
       && (cd "$OUT/ckpt" && sha256sum -c --quiet ../prov1_ckpt.sha256)
     fi \
  && echo "== preserved + verified: $OUT ($(wc -l < "$OUT/prov1_ckpt.sha256") checkpoint hashes recorded; files kept: $KEEP_CKPT)"
}
cleanup() {
  [ "$DONE" = 1 ] || [ -z "$POD" ] && return
  if [ -z "$S" ]; then echo "== cleanup: removing pod $POD (nothing ran on it)"; runpodctl pod remove $POD >/dev/null; return; fi
  $S "cat /workspace/boot.log /workspace/prov1_p*.log 2>/dev/null" > "$OUT/failed_pod.log" 2>/dev/null
  local ev        # three-way evidence check (Sol, SUPPORT-1): anything indeterminate -> stop and KEEP the pod
  ev=$($S 'if [ ! -e /workspace/prov1 ]; then echo NONE; exit 0; fi
           f=$(find /workspace/prov1 \( -name "*.pt" -o -name "*.json" \) -print -quit) || { echo UNKNOWN; exit 0; }
           if [ -n "$f" ]; then echo EVIDENCE; else echo NONE; fi' 2>/dev/null)
  case "$ev" in
    EVIDENCE)
      if preserve; then echo "== cleanup: evidence preserved; removing pod $POD"; runpodctl pod remove $POD >/dev/null
      else echo "== cleanup: evidence could NOT be preserved - pod $POD STOPPED and KEPT (tell Claude; storage bills until removed)"
           runpodctl pod stop $POD >/dev/null; fi;;
    NONE) echo "== cleanup: confirmed no evidence on the pod; removing pod $POD"; runpodctl pod remove $POD >/dev/null;;
    *) echo "== cleanup: could not inspect the pod (${ev:-no answer}) - pod $POD STOPPED and KEPT (tell Claude)"
       runpodctl pod stop $POD >/dev/null;;
  esac
}
trap cleanup EXIT
pid() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("id",""))
except Exception: pass' "$1"; }
for round in $(seq ${CB_ROUNDS:-48}); do
  if runpodctl pod create --name cb-prov1-p$PHASE --template-id runpod-torch-v280 --gpu-id "$GPU_NAME" --cloud-type SECURE \
       --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --min-cuda-version 13.0 \
       --ssh --wait --wait-timeout 20m > ~/cb_prov1_pod.json 2> ~/cb_prov1_pod.err; then
    POD=$(pid ~/cb_prov1_pod.json); [ -n "$POD" ] && break
  fi
  LEFT=$(pid ~/cb_prov1_pod.err); [ -n "$LEFT" ] && { echo "== removing unreachable pod $LEFT"; runpodctl pod remove $LEFT >/dev/null; }
  grep -q 'no longer any instances' ~/cb_prov1_pod.err || { echo "STOP: pod create failed:"; cat ~/cb_prov1_pod.err; exit 1; }
  echo "== $(date +%H:%M) round $round: no $GPU_NAME available (CUDA 13 hosts); retry in 5 min"; sleep 300
done
[ -n "$POD" ] || { echo "STOP: no GPU after ${CB_ROUNDS:-48} rounds"; exit 1; }
echo "== pod $POD ($GPU_NAME) phase $PHASE"
eval "$(runpodctl ssh info $POD | python3 -c 'import sys,json;d=json.load(sys.stdin);print("IP=%s PORT=%s" % (d["ip"], d["port"]))')"
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $K -p $PORT root@$IP"
echo "== boot: run 2's exact environment, model + tokenizer, pure tests (10-15 min)"
$S "cat > /workspace/boot.sh" <<'BOOT' || { echo "STOP: could not upload boot.sh"; exit 1; }
set -Eeuo pipefail
exec > /workspace/boot.log 2>&1
trap 'rc=$?; [ $rc = 0 ] || echo "STOP: boot failed (exit $rc)"' EXIT
git clone -q --depth 400 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D && git checkout -q --detach $PIN && [ "$(git rev-parse HEAD)" = "$PIN" ]
python3 -m venv /workspace/venv || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q /workspace/venv; }
grep -v '^Python ' results/cons0a_run2/ENV.txt | grep -v '^-e ' > /workspace/req.txt
/workspace/venv/bin/python -m pip install -q -r /workspace/req.txt pytest || { echo 'STOP: could not install run 2 environment'; exit 1; }
/workspace/venv/bin/python -c "import torch, transformers, sys; assert torch.cuda.is_available(); print('ENV python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))"
/workspace/venv/bin/python - <<PY || { echo 'STOP: download or identity check failed'; exit 1; }
import hashlib, importlib.util, json
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", revision="$REV", local_dir="/workspace/hf/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
spec = importlib.util.spec_from_file_location("b", "scripts/cls0c_b.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
m = json.load(open("results/cls0b_final/seed2.json"))["model_manifest"]["files"]
assert b.manifest("/workspace/hf/Qwen3-8B") == m, "model files differ from the CLS-0B manifest"
open("/workspace/model_manifest.sha256", "w").write(hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest())
print("BOOT model == CLS-0B manifest")
PY
/workspace/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_support1.py tests/test_prov1.py tests/test_prov1_analyze.py \
  tests/test_prov1_sol_review.py -m "not slow" -k "not end_to_end and not calibration and not cli" > /workspace/pytest.log 2>&1 \
  || { tail -30 /workspace/pytest.log; echo 'STOP: PROV-1 tests fail in this environment'; exit 1; }
tail -1 /workspace/pytest.log
find $D -name __pycache__ -prune -exec rm -rf {} +
echo "BOOT OK"
BOOT
[ "$($S "test -s /workspace/boot.sh && bash -n /workspace/boot.sh && echo UPLOAD_OK" 2>/dev/null)" = UPLOAD_OK ] \
  || { echo "STOP: boot.sh did not upload intact"; exit 1; }
[ "$($S "D=$D PIN=$PIN REV=$REV nohup bash /workspace/boot.sh > /dev/null 2>&1 < /dev/null & sleep 3; test -f /workspace/boot.log && echo STARTED" 2>/dev/null)" = STARTED ] \
  || { echo "STOP: boot did not start (no boot.log)"; exit 1; }
shown=0; t0=$(date +%s); BOOT_DEADLINE=${BOOT_DEADLINE:-2400}
while :; do
  [ $(( $(date +%s) - t0 )) -gt $BOOT_DEADLINE ] && { echo "STOP: boot did not finish in $((BOOT_DEADLINE / 60)) min - see $OUT/failed_pod.log"; exit 1; }
  sleep ${BOOT_POLL:-30}
  R=$($S "grep -E '^(ENV|BOOT|STOP|[0-9]+ passed|FAILED)' /workspace/boot.log 2>/dev/null; echo __POLL_OK__" 2>/dev/null)
  case "$R" in *__POLL_OK__*) ;; *) echo "$(date +%H:%M) (pod not reachable this poll)"; continue;; esac
  L=$(printf '%s\n' "$R" | grep -v '^__POLL_OK__$')
  n=$(printf '%s\n' "$L" | grep -c .); [ "$n" -gt "$shown" ] && { printf '%s\n' "$L" | grep . | tail -n $((n - shown)); shown=$n; }
  printf '%s\n' "$L" | grep -q '^BOOT OK' && break
  printf '%s\n' "$L" | grep -q '^STOP' && { echo "STOP: boot failed - see $OUT/failed_pod.log"; exit 1; }
done

# ------------------------------------------------------------------ the job (runs detached on the pod) -----------------
JOBF=$(mktemp); cat > $JOBF <<'JOB'
cd /workspace/Continuous_brain_prov1
PY=/workspace/venv/bin/python; M=/workspace/hf/Qwen3-8B; MS=$(cat /workspace/model_manifest.sha256); R=/workspace/prov1
X="$PY scripts/prov1.py"; X1="$PY scripts/support1.py"; P0=$PWD/results/prov1/phase0; P1=$PWD/results/prov1/phase1
S0=$PWD/results/support1/phase0; S1=$PWD/results/support1/phase1; mkdir -p $R
fin() { cd /workspace && tar czf prov1_phase$PHASE.tar.gz --exclude='*.pt' prov1 && sha256sum prov1_phase$PHASE.tar.gz > prov1_phase$PHASE.tar.gz.sha256
  echo "BUNDLE $(cat prov1_phase$PHASE.tar.gz.sha256)"; echo "PROV1 P$PHASE DONE exit=$1"; exit $1; }
# ---- phase-0 identity (every phase)
CH=$($X codehash); G=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[ "$(grep '^CODE_HASH ' $P0/PHASE0.sha256 | cut -d' ' -f2)" = "$CH" ] || { echo "STOP: CODE_HASH $CH is not phase 0's"; fin 1; }
[ "$(grep '^MODEL_MANIFEST ' $P0/PHASE0.sha256 | cut -d' ' -f2)" = "$MS" ] || { echo "STOP: model differs from phase 0"; fin 1; }
[ "$(grep '^SUPPORT1_CODE_HASH ' $P0/PHASE0.sha256 | cut -d' ' -f2)" = "$($X1 codehash)" ] || { echo "STOP: SUPPORT-1 code is not frozen"; fin 1; }
[ "$G" = "NVIDIA RTX 6000 Ada Generation" ] || { echo "STOP: GPU '$G' is not SUPPORT-1's RTX 6000 Ada"; fin 1; }
mkdir -p $R/p0 $R/regen && tar xzf $P0/preps.tar.gz -C $R/p0 \
  && (cd $R/p0 && grep '  prov1_prep_seed' $P0/PHASE0.sha256 | sha256sum -c --quiet -) || { echo "STOP: committed preps differ from PHASE0.sha256"; fin 1; }
$X prep --seeds 20,21,22,23,24 --tokenizer $M --model-manifest-sha $MS --out $R/regen > $R/regen/prep.log 2>&1 \
  && (cd $R/regen && grep '  prov1_prep_seed' $P0/PHASE0.sha256 | sha256sum -c --quiet -) || { echo "STOP: preps do not regenerate identically here"; fin 1; }
rm -rf $R/regen
$PY scripts/prov1_sol_validate.py --prep $R/p0/prov1_prep_seed2{0,1,2,3,4}.json --support1-phase0 $S0 --tokenizer $M \
   --model-manifest-sha $MS --out $R/p0/validation.json > $R/p0/validation.log 2>&1
$PY -c "import json,sys; v=json.load(open('$R/p0/validation.json')); sys.exit(0 if v['verdict']=='PASS' and not v['errors'] and v['code_hash']=='$CH' else 1)" \
  || { echo "STOP: Sol's validator did not PASS on this pod"; fin 1; }
echo "== phase 0 verified: code $CH, model $MS, gpu $G, preps identical, validator PASS"
tim() { local t0=$(date +%s.%N); "$@"; local rc=$?; echo "$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.3f", b-a}') $*" >> $R/timings.log; return $rc; }

if [ "$PHASE" = 1 ]; then
  # ---- SUPPORT-1 reproducibility pilot: seed 24, P, sleep 1 vs its committed phase-1 hash (phase A's precondition)
  mkdir -p $R/repro
  $X1 train --prep $S0/prep_seed24.json --model $M --out $R/repro --arms P --sleeps 1 --model-manifest-sha $MS || fin 1
  $PY - <<PY || fin 1
import json
a = json.load(open("$R/repro/train_seed24.json"))["arms"]["P"]["1"]; b = json.load(open("$S1/cal/train_seed24.json"))["arms"]["P"]["1"]
r = {"regenerated": a["ckpt_sha256"], "committed": b["ckpt_sha256"], "file_equal": a["ckpt_sha256"] == b["ckpt_sha256"],
     "full_state_equal": a["full_state_hash"] == b["full_state_hash"]}
json.dump(r, open("$R/repro/repro.json", "w"), indent=1); print("REPRO", r)
PY
  if $PY -c "import json,sys; sys.exit(0 if json.load(open('$R/repro/repro.json'))['file_equal'] is True else 1)"; then
    rm -f $R/repro/*.pt                                   # matching: reproducible, already on record
  else
    echo "REPRO mismatch: pilot checkpoint kept for preservation and investigation"   # Sol: never delete non-matching evidence
  fi
  # ---- PROV-1 calibration: RP and RC, seed 24, sleep 1; twins + base; timings
  C=$R/cal; mkdir -p $C
  tim $X train --prep $R/p0/prov1_prep_seed24.json --model $M --out $C --arms RP,RC --sleeps 1 --model-manifest-sha $MS || fin 1
  for arm in RP RC; do for mode in merged unmerged; do
    tim $X evalck --prep $R/p0/prov1_prep_seed24.json --model $M --sleep 1 --arm $arm --ckpt $C/cortex_${arm}_s1.pt --mode $mode \
        --monitors --out $C/eval_${arm}_s1_$mode.json --model-manifest-sha $MS || fin 1
  done; done
  tim $X evalck --prep $R/p0/prov1_prep_seed24.json --model $M --sleep 1 --arm N --monitors --out $C/eval_N_s1.json --model-manifest-sha $MS || fin 1
  $PY - <<PY || fin 1
import json, glob
tr = json.load(open("$C/prov1_train_seed24.json"))
items = sum(v["1"]["budget"][0] for v in tr["arms"].values()); secs = sum(v["1"]["seconds"] for v in tr["arms"].values())
rows = sum(len(json.load(open(f))["rows"]) for f in glob.glob("$C/eval_*.json"))
esec = sum(float(l.split()[0]) for l in open("$R/timings.log") if " evalck " in l)
t = {"train_items": items, "train_seconds": secs, "eval_rows": rows, "eval_seconds": esec}
json.dump(t, open("$R/timings.json", "w"), indent=1); print("TIMINGS", t)
PY
  $X calcheck --prep $R/p0/prov1_prep_seed24.json --train $C/prov1_train_seed24.json --rp-merged $C/eval_RP_s1_merged.json \
     --rp-unmerged $C/eval_RP_s1_unmerged.json --rc-merged $C/eval_RC_s1_merged.json --rc-unmerged $C/eval_RC_s1_unmerged.json \
     --base $C/eval_N_s1.json --out $C/calcheck.json; CC=$?
  $X project --timings $R/timings.json --preps $(echo $R/p0/prov1_prep_seed2{0,1,2,3}.json | tr ' ' ,) \
     --support1-preps $(echo $S0/prep_seed2{0,1,2,3}.json | tr ' ' ,) --usd-per-hour $USD_PER_HR --spent ${SPENT:-0} --ceiling 30 \
     --out $R/projection.json; PJ=$?
  rm -f $C/*.pt
  (cd $R && sha256sum cal/calcheck.json cal/prov1_train_seed24.json cal/eval_RP_s1_merged.json cal/eval_RP_s1_unmerged.json \
       cal/eval_RC_s1_merged.json cal/eval_RC_s1_unmerged.json cal/eval_N_s1.json timings.json timings.log projection.json \
       repro/repro.json repro/train_seed24.json p0/validation.json > PHASE1.sha256 \
     && printf 'PIN %s\nCODE_HASH %s\nMODEL_MANIFEST %s\nGPU %s\n' "$(git -C /workspace/Continuous_brain_prov1 rev-parse HEAD)" "$CH" "$MS" "$G" >> PHASE1.sha256)
  [ $CC = 0 ] || fin 4
  [ $PJ = 0 ] || fin 5
  fin 0
fi

# ---- phases A and B: phase 1 must be committed, verified and passing
(cd $P1 && grep -v '^PIN \|^CODE_HASH \|^MODEL_MANIFEST \|^GPU ' PHASE1.sha256 | sha256sum -c --quiet -) || { echo "STOP: phase-1 artifacts differ from PHASE1.sha256"; fin 1; }
[ "$(grep '^CODE_HASH ' $P1/PHASE1.sha256 | cut -d' ' -f2)" = "$CH" ] && [ "$(grep '^GPU ' $P1/PHASE1.sha256 | cut -d' ' -f2-)" = "$G" ] \
  || { echo "STOP: code or GPU differs from phase 1"; fin 1; }
$PY -c "import json,sys; c=json.load(open('$P1/cal/calcheck.json')); p=json.load(open('$P1/projection.json')); r=json.load(open('$P1/repro/repro.json')); sys.exit(0 if c['pass'] is True and not c['errors'] and p['phase_A_affordable'] and ('$PHASE'=='B' or r['file_equal']) else 1)" \
  || { echo "STOP: calibration, projection or (phase A) reproducibility pilot did not pass"; fin 1; }

if [ "$PHASE" = A ]; then
  REF=$R/s1ref; mkdir -p $REF && tar xzf $PWD/results/support1/phase2/support1_phase2.tar.gz -C $REF \
    && [ "$(sha256sum < $PWD/results/support1/phase2/support1_phase2.tar.gz | cut -d' ' -f1)" = "$(cut -d' ' -f1 $PWD/results/support1/phase2/BUNDLE.sha256)" ] \
    || { echo "STOP: SUPPORT-1 phase-2 bundle missing or altered"; fin 1; }
  for s in 20 21 22 23; do
    G1=$R/regen1/seed$s; O=$R/seed$s; mkdir -p $G1 $O; cp $R/p0/prov1_prep_seed$s.json $O/
    echo "== seed $s regenerate SUPPORT-1 $(date -u +%FT%TZ)"
    $X1 train --prep $S0/prep_seed$s.json --model $M --out $G1 --fork --model-manifest-sha $MS --validator-report $S0/validation.json \
       --calibration-result $S1/cal/calcheck.json --calibration-prep $S0/prep_seed24.json || fin 1
    $PY - <<PY || { echo "STOP: seed $s checkpoints are not SUPPORT-1's (reuse stopped pending investigation or verified upload)"; fin 6; }
import hashlib, json, sys
new = json.load(open("$G1/train_seed$s.json")); old = json.load(open("$REF/support1/seed$s/train_seed$s.json"))
rec, bad = {}, []
def h(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
for arm in ("P", "U4"):
    for L in range(1, 6):
        f = "$G1/cortex_%s_s%d.pt" % (arm, L); want = old["arms"][arm][str(L)]["ckpt_sha256"]
        rec["%s|%s" % (arm, L)] = {"logged_sha256": want, "regenerated_sha256": h(f), "full_state_equal":
                                   new["arms"][arm][str(L)]["full_state_hash"] == old["arms"][arm][str(L)]["full_state_hash"]}
f = "$G1/cortex_U4_s4_withheld.pt"; want = old["fork"]["withheld"]["U4"]["ckpt_sha256"]
rec["U4|4w"] = {"logged_sha256": want, "regenerated_sha256": h(f),
                "full_state_equal": new["fork"]["withheld"]["U4"]["full_state_hash"] == old["fork"]["withheld"]["U4"]["full_state_hash"]}
for k, v in rec.items():
    v["admitted"] = v["regenerated_sha256"] == v["logged_sha256"]
    if not v["admitted"]: bad.append(k)
json.dump({"seed": $s, "records": rec, "all_admitted": not bad}, open("$O/admission.json", "w"), indent=1)
print("ADMIT seed $s", "44-set part:", len(rec), "mismatches:", bad); sys.exit(1 if bad else 0)
PY
    for arm in P U4; do for L in 1 2 3 4 5; do for mode in merged unmerged; do
      tim $X evalck --prep $O/prov1_prep_seed$s.json --model $M --sleep $L --arm $arm --ckpt $G1/cortex_${arm}_s$L.pt --mode $mode \
          --conditions FAITHFUL,DEL-Q,DEL-X,ADD,FULL --source support1 --out $O/eval_${arm}_s${L}_$mode.json --model-manifest-sha $MS || fin 1
    done; done; done
    for mode in merged unmerged; do
      tim $X evalck --prep $O/prov1_prep_seed$s.json --model $M --sleep 4 --arm U4 --ckpt $G1/cortex_U4_s4_withheld.pt --mode $mode \
          --history withheld --conditions FAITHFUL,DEL-Q,DEL-X,ADD,FULL --source support1 --out $O/eval_U4_s4_withheld_$mode.json \
          --model-manifest-sha $MS || fin 1
    done
    find $G1 -name '*.pt' -delete                         # identical by sha256 to SUPPORT-1's preserved copies on E:
  done
  echo "== analysis $(date -u +%FT%TZ)"
  $PY scripts/prov1_analyze.py --phase A --dir $R --support1-dir $REF/support1 --out $R/analysis_claude.json || fin 1
  $PY scripts/prov1_sol_analyze.py --phase A --dir $R --support1-dir $REF/support1 --out $R/analysis_sol.json; echo "SOL ANALYZER rc=$?"
  grep -h '"verdict"' $R/analysis_claude.json $R/analysis_sol.json | head -2
  fin 0
fi

if [ "$PHASE" = B ]; then
  REF=$R/s1ref; mkdir -p $REF && tar xzf $PWD/results/support1/phase2/support1_phase2.tar.gz -C $REF || { echo "STOP: SUPPORT-1 bundle"; fin 1; }
  [ -s $PWD/results/prov1/phaseA/PHASEA.sha256 ] || { echo "STOP: phase A not committed (B follows the phase-A review)"; fin 1; }
  for s in 20 21 22 23; do
    O=$R/seed$s; mkdir -p $O; cp $R/p0/prov1_prep_seed$s.json $O/
    echo "== seed $s train $(date -u +%FT%TZ)"
    $X train --prep $O/prov1_prep_seed$s.json --model $M --out $O --fork --model-manifest-sha $MS --validator-report $R/p0/validation.json \
       --calibration-result $P1/cal/calcheck.json --calibration-prep $R/p0/prov1_prep_seed24.json || fin 1
    for L in 1 2 3 4 5; do
      $X evalck --prep $O/prov1_prep_seed$s.json --model $M --sleep $L --arm N --monitors --out $O/eval_N_s$L.json --model-manifest-sha $MS || fin 1
      for arm in RP RU RC; do for mode in merged unmerged; do
        $X evalck --prep $O/prov1_prep_seed$s.json --model $M --sleep $L --arm $arm --ckpt $O/cortex_${arm}_s$L.pt --mode $mode \
           --monitors --out $O/eval_${arm}_s${L}_$mode.json --model-manifest-sha $MS || fin 1
      done; done
    done
    for arm in RU RC; do for mode in merged unmerged; do
      $X evalck --prep $O/prov1_prep_seed$s.json --model $M --sleep 4 --arm $arm --ckpt $O/cortex_${arm}_s4_withheld.pt --mode $mode \
         --history withheld --monitors --out $O/eval_${arm}_s4_withheld_$mode.json --model-manifest-sha $MS || fin 1
    done; done
  done
  echo "== analysis $(date -u +%FT%TZ)"
  $PY scripts/prov1_analyze.py --phase B --dir $R --support1-dir $REF/support1 --out $R/analysis_claude.json || fin 1
  $PY scripts/prov1_sol_analyze.py --phase B --dir $R --support1-dir $REF/support1 --out $R/analysis_sol.json; echo "SOL ANALYZER rc=$?"
  grep -h '"verdict"' $R/analysis_claude.json $R/analysis_sol.json | head -2
  fin 0
fi
JOB
sed -i "1i PHASE=$PHASE; USD_PER_HR=$USD_PER_HR; SPENT=${SPENT:-0}" $JOBF   # these travel inside the job file
JSHA=$(sha256sum < $JOBF | cut -d' ' -f1); up=0
for i in 1 2 3; do
  $S "cat > /workspace/prov1_p$PHASE.sh" < $JOBF 2>/dev/null
  [ "$($S "sha256sum < /workspace/prov1_p$PHASE.sh 2>/dev/null | cut -d' ' -f1" 2>/dev/null)" = "$JSHA" ] && { up=1; break; }
  echo "== job upload attempt $i not verified; retrying"; sleep ${RETRY_WAIT:-20}
done
[ $up = 1 ] || { echo "STOP: job script could not be uploaded intact"; exit 1; }
st=""
for i in 1 2 3; do       # exactly-once start (atomic mkdir marker), as SUPPORT-1
  st=$($S "if mkdir /workspace/prov1_p$PHASE.started 2>/dev/null; then (nohup bash /workspace/prov1_p$PHASE.sh > /workspace/prov1_p$PHASE.log 2>&1 < /dev/null &); sleep 3; [ -e /workspace/prov1_p$PHASE.log ] && echo STARTED; else echo EXISTS; fi" 2>/dev/null)
  case "$st" in STARTED|EXISTS) break;; esac
  echo "== job start attempt $i not confirmed; retrying"; sleep ${RETRY_WAIT:-20}
done
case "$st" in STARTED|EXISTS) echo "== job started";; *) echo "STOP: job did not start"; exit 1;; esac
t0=$(date +%s)
while :; do
  [ $(( $(date +%s) - t0 )) -gt $(( CAP_H * 3600 )) ] && { echo "== ${CAP_H}-hour cap reached"; exit 1; }
  sleep ${JOB_POLL:-120}
  R=$($S "if [ ! -e /workspace/prov1_p$PHASE.log ]; then echo NOLOG; elif grep -q 'PROV1 P$PHASE DONE' /workspace/prov1_p$PHASE.log; then echo DONE; else echo RUNNING; fi" 2>/dev/null)
  case "$R" in
    DONE) echo "== done"; break;;
    NOLOG) nolog=$(( ${nolog:-0} + 1 )); [ $nolog -ge 3 ] && { echo "STOP: the job log is missing on a reachable pod"; exit 1; };;
    RUNNING) nolog=0;;
    *) echo "$(date +%H:%M) (pod not reachable this poll)"; continue;;
  esac
  echo "$(date +%H:%M) $($S 2>/dev/null "grep -E '^(==|EVAL|TRAIN|STOP|verdict|SOL|CALCHECK|PROJECTION|REPRO|ADMIT|TIMINGS)' /workspace/prov1_p$PHASE.log | tail -1 | cut -c1-150")"
done
$S "grep -vE 'Loading weights|Writing model' /workspace/prov1_p$PHASE.log" | tail -40
$S "cat /workspace/prov1_phase$PHASE.tar.gz.sha256" > "$OUT/prov1_phase$PHASE.tar.gz.sha256" \
  && $S "cat /workspace/prov1_phase$PHASE.tar.gz" > "$OUT/prov1_phase$PHASE.tar.gz" \
  && $S "cat /workspace/prov1_p$PHASE.log /workspace/boot.log" > "$OUT/prov1_p$PHASE.log" \
  && (cd "$OUT" && sha256sum -c prov1_phase$PHASE.tar.gz.sha256) && preserve \
  && echo "== downloaded + verified: $OUT/prov1_phase$PHASE.tar.gz" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] && { cp "$OUT/prov1_phase$PHASE.tar.gz" $DL/ 2>/dev/null && echo "== copy in Downloads" || echo "== (no copy in Downloads; the E: copy is the record)"; }
[ "$DONE" = 1 ] || { echo "== download did not complete"; exit 1; }
EX=$(grep -o "PROV1 P$PHASE DONE exit=[0-9]*" "$OUT/prov1_p$PHASE.log" | tail -1 | cut -d= -f2); echo "== PROV-1 phase $PHASE exit: ${EX:-UNKNOWN}"
case "$EX" in 0) exit 0;; 4) exit 4;; 5) exit 5;; 6) exit 6;; *) exit 1;; esac
