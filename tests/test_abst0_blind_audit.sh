#!/usr/bin/env bash
# Blind-audit tooling: export hides identity, bind satisfies validate_audit, compare counts correctly, refusals hold.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
B="python3 $R/scripts/abst0_blind_audit.py"
python3 - "$T" <<'PY'
import json, sys; from pathlib import Path
d = Path(sys.argv[1])
for s in (12, 13):
    items = [{"sleep": L, "entity": f"E{s}{L}{i}", "memory": f"mem {s} {L} {i}", "q": f"q {s} {L} {i}?"}
             for L in range(1, 6) for i in range(3)]
    (d / f"audit_sample_seed{s}.json").write_text(json.dumps({"seed": s, "items": items}))
    (d / f"prep_seed{s}.json").write_text(json.dumps({"seed": s}))
PY
$B export --dir $T --seeds 12,13 --secret s3cr3t --blind $T/blind.json --key $T/key.json >/dev/null
python3 - "$T" <<'PY'
import json, sys; from pathlib import Path
d = Path(sys.argv[1]); b = json.loads((d/"blind.json").read_text()); k = json.loads((d/"key.json").read_text())
assert len(b["items"]) == 30 and all(set(x) == {"audit_id", "memory", "question"} for x in b["items"])
assert [x["memory"] for x in b["items"]] != sorted(x["memory"] for x in b["items"]), "not shuffled"
for name, flip in (("A", set()), ("B", {"a0000", "a0001", "a0002"})):
    L = {x["audit_id"]: {"label": ("supported" if x["audit_id"] in flip else "unsupported"), "rationale": "r"}
         for x in b["items"]}
    (d/f"{name}.json").write_text(json.dumps({"auditor": name, "frozen": True, "blind_sha256": k["blind_sha256"], "labels": L}))
L = {x["audit_id"]: "unsupported" for x in b["items"]}
(d/"unfrozen.json").write_text(json.dumps({"auditor": "X", "blind_sha256": k["blind_sha256"], "labels": L}))
(d/"wrongbind.json").write_text(json.dumps({"auditor": "X", "frozen": True, "blind_sha256": "0"*64, "labels": L}))
L2 = dict(L); L2.pop("a0005")
(d/"missing.json").write_text(json.dumps({"auditor": "X", "frozen": True, "blind_sha256": k["blind_sha256"], "labels": L2}))
PY
mkdir $T/out
# Sol's mapping attack: swap two key entries -> bind must refuse rather than silently exchange labels
python3 -c "import json;k=json.load(open('$T/key.json'));m=k['map'];m['a0000'],m['a0001']=m['a0001'],m['a0000'];json.dump(k,open('$T/swapped_key.json','w'))"
if $B bind --dir $T --key $T/swapped_key.json --labels $T/A.json --out-dir $T/out --blind $T/blind.json 2>/dev/null; then echo "FAIL: swapped key accepted"; exit 1; fi
$B bind --dir $T --key $T/key.json --labels $T/A.json --out-dir $T/out --blind $T/blind.json >/dev/null
python3 - "$T" "$R" <<'PY'
import json, sys, hashlib; from pathlib import Path
d = Path(sys.argv[1]); sys.path.insert(0, sys.argv[2] + "/scripts"); from abst0 import validate_audit
for s in (12, 13):
    smp = json.loads((d/f"audit_sample_seed{s}.json").read_text()); lab = json.loads((d/"out"/f"audit_labels_seed{s}.json").read_text())
    assert validate_audit(hashlib.sha256((d/f"prep_seed{s}.json").read_bytes()).hexdigest(), smp, lab) == []
    assert [l["q"] for l in lab["labels"]] == [x["q"] for x in smp["items"]]
PY
$B compare --key $T/key.json --a $T/A.json --b $T/B.json --out $T/cmp.json >/dev/null
python3 -c "import json;r=json.load(open('$T/cmp.json'));assert r['n']==30 and r['table']['unsupported']['supported']==3 and abs(r['raw_agreement']-0.9)<1e-12 and r['disagreements']==['a0000','a0001','a0002'], r"
for bad in unfrozen wrongbind missing; do
  if $B bind --dir $T --key $T/key.json --labels $T/$bad.json --out-dir $T/out --blind $T/blind.json 2>/dev/null; then echo "FAIL: $bad accepted"; exit 1; fi
done
python3 -c "import json;p='$T/audit_sample_seed12.json';s=json.load(open(p));s['items'][0]['q']='changed?';json.dump(s,open(p,'w'))"
if $B bind --dir $T --key $T/key.json --labels $T/A.json --out-dir $T/out --blind $T/blind.json 2>/dev/null; then echo "FAIL: changed sample accepted"; exit 1; fi
echo "PASS: ABST-0 blind audit (export hides identity + shuffles, bind passes validate_audit in order, compare table/agreement, refuses a swapped key entry, unfrozen / wrong-bound / incomplete labels and a changed sample)"
