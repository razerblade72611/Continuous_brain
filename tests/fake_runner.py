"""Drive the REAL scripts/run_v2.py with a fake model and fake seeds (tests only; no training, and no data file is read).

  python -I tests/fake_runner.py <repo root> sealed [--resume] [--max-seeds N]
Env: CRASH_AT=<seed> kill mid-seed; CRASH_FIRST_WRITE=1 kill before the initial state write;
     CRASH_BEFORE_LOG=<seed> kill after that seed's row is written but before its marker entry;
     SLEEP=<s> per seed (to hold the lock).
Each fake row carries a random nonce, so a rerun of a recorded seed would be detectable.
"""
import os
import sys
import time

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/scripts")
import being.gate_v2 as g  # noqa: E402

g.pool_ids = lambda tok: list(range(80))
g.load_sentences = lambda name: ["x"]
g.mean_loss = lambda m, t, s: 1.0
g.backbone_hash = lambda m: "H"
g.summarize = lambda rows, cfg: {"checks": {"stub": True}, "backbone_unattested_seeds": [], "n": len(rows)}


def fake_seed(model, tok, cfg, s, *a):
    time.sleep(float(os.environ.get("SLEEP", "0")))
    if os.environ.get("CRASH_AT") == str(s):
        os._exit(137)
    return {"seed": s, "backbone_sha256": "H", "backbone_sha256_end": "H", "nonce": os.urandom(8).hex()}


g.run_seed = fake_seed
import run_v2  # noqa: E402

run_v2.load_backbone = lambda p: (None, None)
_write, _log = run_v2.write_atomic, run_v2.log_marker


def write(path, obj):
    if os.environ.get("CRASH_FIRST_WRITE"):
        os._exit(137)
    _write(path, obj)


def log(entry):
    if str(entry.get("row_recorded")) == os.environ.get("CRASH_BEFORE_LOG"):
        os._exit(137)
    _log(entry)


run_v2.write_atomic, run_v2.log_marker = write, log
sys.argv = ["run_v2.py"] + sys.argv[2:]
run_v2.main()
