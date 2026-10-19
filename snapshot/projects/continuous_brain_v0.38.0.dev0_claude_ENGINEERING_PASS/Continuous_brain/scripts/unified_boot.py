#!/usr/bin/env python3
"""Boot the Unified Individual runtime, bind what the legacy manifest verifies, and report.

With no legacy root, every mechanism slot is UNBOUND (the honest state until
DAV-74). With --legacy-root, every VERIFIED manifest entry is re-hashed under
that root; a mismatch refuses to run unless --dev-override is given (which is
logged and makes the runtime non-scientific). The runtime then ticks with the
coprocessor disabled and saves a never-overwrite checkpoint.

Examples (inside WSL, repo root):
  python scripts/unified_boot.py --out runs/unified_boot
  python scripts/unified_boot.py --legacy-root ~/projects/continuous_brain_v0.12.2 --out runs/unified_boot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuous_brain_snn.snapshot import SnapshotStore  # noqa: E402
from continuous_brain_unified import RuntimeConfig, UnifiedRuntime  # noqa: E402
from continuous_brain_unified.legacy import (LegacyManifest, build_legacy_components,  # noqa: E402
                                             resolve_legacy_root)

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-root", default=None)
    ap.add_argument("--manifest", default=str(REPO / "legacy" / "manifest.json"))
    ap.add_argument("--dev-override", action="store_true", help="bind despite hash mismatches (logged, non-scientific)")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--out", default="runs/unified_boot")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rt = UnifiedRuntime(RuntimeConfig(), log_dir=out / "logs")
    manifest = LegacyManifest.load(args.manifest)
    if manifest.entries:
        root = resolve_legacy_root(args.legacy_root)
        outcomes = build_legacy_components(manifest, root, args.dev_override, out / "logs" / "dev_overrides.jsonl")
        rt.bind_many([o.component for o in outcomes if o.bound])       # two bound entries for one slot: refused
        reasons: dict[str, list[str]] = {}
        for o in outcomes:
            if not o.bound and o.slot not in rt.components:
                reasons.setdefault(o.slot, []).append(o.reason)
        for slot_name, why in reasons.items():
            rt.mark_unbound(slot_name, "; ".join(why))
    rt.run(args.ticks)
    snap = rt.save(SnapshotStore(out), label="unified_boot")
    report = {"scientific": rt.scientific, "ticks": rt.tick + 1, "checkpoint": str(snap),
              "binding": [asdict(r) for r in rt.binding_report()], "manifest_sha256": manifest.source_sha256}
    (out / "binding_report.json").write_text(json.dumps(report, indent=1))
    for r in report["binding"]:
        print(f"{r['slot']:<16} {r['status']:<15} {r['component'] or '':<20} {r['reason']}")
    print(f"checkpoint: {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
