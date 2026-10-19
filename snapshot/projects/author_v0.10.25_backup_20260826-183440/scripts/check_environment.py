from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run(command: list[str], env: dict[str, str] | None = None):
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        return process.returncode, (process.stdout + process.stderr).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("WSL:", "yes" if "microsoft" in platform.release().lower() else "no")
print("DISPLAY:", os.getenv("DISPLAY", ""))
print("WAYLAND_DISPLAY:", os.getenv("WAYLAND_DISPLAY", ""))
print("/dev/dxg:", "present" if Path("/dev/dxg").exists() else "missing")

for executable in ("glxinfo", "vulkaninfo", "llama-server", "llama-cli"):
    print(f"{executable}:", shutil.which(executable) or "not found")

if shutil.which("glxinfo"):
    env = os.environ.copy()
    env.setdefault("GALLIUM_DRIVER", "d3d12")
    env.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "AMD")
    _, output = run(["glxinfo", "-B"], env=env)
    print("\nOpenGL / WSLg summary:")
    for line in output.splitlines():
        if any(key in line for key in ("Vendor:", "Device:", "Accelerated:", "OpenGL renderer")):
            print(line)

if shutil.which("vulkaninfo"):
    _, output = run(["vulkaninfo", "--summary"])
    print("\nVulkan summary (informational; WSL AI2-THOR uses GLX in the tested setup):")
    print(output[:8000] if output else "(unavailable)")
