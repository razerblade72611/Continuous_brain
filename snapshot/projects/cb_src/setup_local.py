"""Standard-library launcher: use a working env or install into a private venv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def environment_ready(python, cpu=False):
    script = (
        "import importlib.metadata as m; import torch,numpy,PIL; "
        "assert torch.__version__.split('+')[0]=='2.4.1'; "
        + ("" if cpu else "import torch_directml; assert m.version('torch-directml')=='0.2.5.dev240914'; ")
        + "print('ready')"
    )
    try:
        result = subprocess.run([str(python), "-c", script], capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Set up and launch Continuous Brain locally")
    p.add_argument("--existing", action="store_true", help="Require this already-working Python environment")
    p.add_argument("--cpu", action="store_true", help="Explicit CPU diagnostic installation")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        print("Use 64-bit Python 3.10, 3.11, or 3.12; the pinned DirectML wheel requires one of these.")
        return 1
    import struct
    if struct.calcsize("P") != 8:
        print("Use 64-bit Python.")
        return 1
    # No package import occurs before source verification.
    verify = subprocess.run([sys.executable, str(ROOT / "verify_release.py")], cwd=ROOT)
    if verify.returncode:
        return verify.returncode
    python = Path(sys.executable)
    ready = environment_ready(python, args.cpu)
    if args.existing and not ready:
        print("This environment is not ready. Run without --existing to create an isolated environment.")
        return 1
    if ready:
        print("Using your existing compatible environment: " + str(python), flush=True)
    else:
        envdir = ROOT / (".venv_cpu" if args.cpu else ".venv_dml")
        python = envdir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            print("Creating a private Python environment. Your other environments are unchanged.", flush=True)
            venv.EnvBuilder(with_pip=True).create(envdir)
        if not environment_ready(python, args.cpu):
            print("Installing dependencies. Internet is needed for this first setup.", flush=True)
            packages = ["numpy==1.26.4", "Pillow>=10.4,<13"]
            if args.cpu:
                packages += ["torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cpu",
                             "--extra-index-url", "https://pypi.org/simple"]
            else:
                packages += ["torch-directml==0.2.5.dev240914", "torch==2.4.1"]
            installed = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", *packages], cwd=ROOT)
            if installed.returncode:
                print("Dependency installation stopped. Check the error above, then launch again to retry.")
                return installed.returncode
            if not environment_ready(python, args.cpu):
                print("Installed environment did not pass its import/version check.")
                return 1
    command = args.command or ["run"]
    if command[0] in ("run", "doctor", "bootstrap", "legacy") and "--device" not in command:
        command += ["--device", "cpu" if args.cpu else "dml"]
    print("Starting: " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen([str(python), "-m", "continuous_brain.local_cli", *command], cwd=ROOT, env=env)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Console Ctrl+C also reaches the child. Let it finish the atomic save.
        print("Waiting for the individual to save. Keep this window open.", flush=True)
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                print("Save is still finishing.", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print("Setup stopped: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
