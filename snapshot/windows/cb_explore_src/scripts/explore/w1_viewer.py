#!/usr/bin/env python3
"""Watch the W1 individual live in a browser (exploration lane, ENGINEERING_ONLY).

usage:
  python scripts/explore/w1_viewer.py [--seed 71001] [--port 8765] [--play]
  python scripts/explore/w1_viewer.py --language hf --model Qwen/Qwen2.5-0.5B-Instruct \
      --local-dir ~/cb_models/Qwen--Qwen2.5-0.5B-Instruct
  python scripts/explore/w1_viewer.py --resume latest        (the individual you watched last, where it was)
  python scripts/explore/w1_viewer.py --skills routine       (the fixed routine library, no learning, one life)
  python scripts/explore/w1_viewer.py --adopt runs/explore/lifelong/run1/individual/checkpoints/<snapshot>
                                                             (watch an individual trained headless, in a new world)
  python scripts/explore/w1_viewer.py --individual cb2-ind-0001
                                                             (the canonical CB2 individual lives its next lives here,
                                                              every tick archived, exactly as cb2_collect.py wake)

By default the individual learns from its lives and continues across them: when it dies, it is saved to disk
automatically and (with "Next life when it dies" checked) wakes as the same individual in the next world.

Then open http://localhost:8765 (from Windows too: WSL forwards localhost).  Ctrl+C stops the server.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: threads only cost
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=71001, help="world seed of the first life")
    ap.add_argument("--perception", choices=("organ", "oracle"), default="organ")
    ap.add_argument("--memory", choices=("full", "none"), default="full")
    ap.add_argument("--listen", choices=("organ", "none"), default="organ", help="none: it ignores the caretaker")
    ap.add_argument("--skills", choices=("learned", "routine"), default="learned",
                    help="learned: foresight and choice learned from its own lives (default); routine: fixed")
    ap.add_argument("--language", choices=("dev", "hf"), default="dev",
                    help="dev: the template stand-in (no weights); hf: an open-weight language model")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--local-dir", default=None, help="the model's directory (from scripts/explore/fetch_models.py)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--resume", default=None, help="'latest', or a snapshot directory written by Save to disk")
    ap.add_argument("--adopt", default=None,
                    help="a snapshot written by scripts/explore/w1_lifelong.py train: watch that trained individual")
    ap.add_argument("--individual", default=None,
                    help="a CB2 individual under runs/explore/cb2/individuals: it lives its own next lives here")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--play", action="store_true", help="start playing immediately")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from continuous_brain_organs.vision import VisionOrgan
    from continuous_brain_individual.viewer import LiveSession, latest_snapshot, serve

    language = {"backend": "dev"}
    if a.language == "hf":
        language = {"backend": "hf", "model_id": a.model, "local_dir": a.local_dir, "device": a.device,
                    **({"dtype": a.dtype} if a.dtype else {})}
    vision = VisionOrgan(WEIGHTS, CARD) if WEIGHTS.exists() else None
    if vision is None and a.perception == "organ" and not a.resume:
        print(f"The vision organ's weights are missing ({WEIGHTS}). Copy tile_decoder_v1.pt there, "
              "or run with --perception oracle.", file=sys.stderr)
        return 2
    root = REPO / "runs/explore/viewer/lives"
    session = LiveSession(vision, language, root)
    if a.individual:
        session.live_as(REPO / "runs/explore/cb2/individuals" / a.individual, REPO / "runs/explore/cb2/archive")
        print(f"Living {a.individual}: life {session.ind.life + 1}, world {session.seed}, every tick archived",
              flush=True)
    elif a.adopt:
        session.adopt(a.adopt, a.seed)
        print(f"Adopted {a.adopt}", flush=True)
    elif a.resume:
        snap = latest_snapshot(root) if a.resume == "latest" else Path(a.resume)
        if snap is None:
            print(f"Nothing saved yet under {root}.", file=sys.stderr)
            return 2
        session.resume(snap)
        print(f"Resumed {snap}", flush=True)
    else:
        session.new_life(a.seed, {"perception": a.perception, "memory": a.memory, "language": a.listen,
                                  "skills": a.skills, "lifelong": a.skills == "learned"})
    httpd, runner = serve(session, a.host, a.port, play=a.play)
    print(f"Watching the individual at http://localhost:{a.port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop_flag = True
        httpd.server_close()
        runner.join(timeout=5)
        if session.lifelong and session.ind is not None:
            print(f"Saved the individual to {session.save()}", flush=True)     # nothing it learned is lost
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
