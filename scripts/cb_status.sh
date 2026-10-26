#!/usr/bin/env bash
# Read-only status of the Continuous Brain workspace on a pod: volume, code, running jobs, A0 and A2 Phase-1 results.
#   bash scripts/cb_status.sh
cd "$(dirname "$0")/.." || exit 1
echo "== volume:  $(findmnt -no SOURCE /workspace 2>/dev/null)"
echo "== commit:  $(git log --oneline -1)"
echo "== running: $(pgrep -af 'stage_v3|run_v3_dev' | head -3 || true)"; pgrep -f stage_v3 >/dev/null || echo "   nothing running"
echo "== files:"; ls -la stageA0.log stageA1run2.log results/dev3/*summary*.json results/dev3/stageA2_A0_describe.* results_*.tar.gz* 2>&1 | sed 's/^/   /'
echo "== A0 verdict:"; grep -m1 "A0 VERDICT" stageA0.log 2>&1 | cut -c1-200 | sed 's/^/   /'
echo "== A0 describe:"
if [ -f results/dev3/stageA2_A0_describe.txt ]; then sed 's/^/   /' results/dev3/stageA2_A0_describe.txt
elif [ -d results/dev3/stageA2_A0 ]; then echo "   (not yet produced - generating now, read-only on the A0 results)"; .venv/bin/python scripts/a0_describe.py | tee results/dev3/stageA2_A0_describe.txt | sed 's/^/   /'
else echo "   no A0 results on this volume"; fi
echo "== Phase 1 run 2 log:"; grep -E "BACKBONE|STAGE|FAILED|BUNDLE|POD STOP|NOT stopping|set aside" stageA1run2.log 2>&1 | cut -c1-240 | sed 's/^/   /'
echo "== Phase 1 run 2 results:"
python3 - <<'PY'
import glob, json
fs = [f for f in sorted(glob.glob("results/dev3/stageA2_A1run2/*.json")) if ".interrupted." not in f and ".mismatch." not in f]
print(f"   {len(fs)} result files, statuses: {sorted({json.load(open(f)).get('status') for f in fs})}")
try:
    s = json.load(open("results/dev3/stageA2_A1run2_summary.json"))
except Exception as e:  # noqa: BLE001
    print("   no summary:", type(e).__name__); raise SystemExit
print(f"   hours {s.get('hours')} | only {s.get('only')} | failed {s.get('failed')}")
for arm, ph in s.get("phases", {}).get("A1", {}).items():
    print(f"   {arm}: best={ph['best']}  rule={ph['rule']}")
    for n, sc in ph["scores"].items():
        print(f"      {n[:50]:50s} acq {sc['acquisition']:.1f} fc_last {sc['fc_last']:.1f} recall_last {sc['recall_last']:.1f}"
              f" margin {sc['margin_last']:.2f} band {sc['band']} ppl {sc['ppl_ratio']}")
PY
