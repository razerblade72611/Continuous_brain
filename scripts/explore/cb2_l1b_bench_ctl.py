#!/usr/bin/env python3
"""CB2-L1b benchmark pod controller (WSL side): launch + watch, with billing safety first (Sol's review S-014).

  launch   verify the package; refuse a duplicate claim; create ONE pod named cb2-l1b-bench-claude (or adopt an
           existing pod only if its live name AND our recorded claim match); from the moment it exists, any failure
           before hand-off deletes it and VERIFIES it is gone; upload with bounded timeouts; start the watcher, then
           PHASE=bench; confirm the bench started.
  watch    every POLL s: re-verify the pod's identity; billing guard first (past MAXH hours: STOP the pod -- storage
           kept -- before any slow transfer); fetch status/log with bounded timeouts; when the bench finishes, download
           cb2_l1b_bench.tgz and verify its sha256 against the pod's, then DELETE and verify deletion.  Bench failure:
           save logs, delete, verify.  Transfer/checksum failure: STOP the pod (evidence kept on its volume) and record.
           A delete or stop that cannot be verified after bounded retries is written as URGENT and exits non-zero.

Exit codes: 0 done and pod deleted | 2 URGENT (cleanup unverified: the pod may still bill) | 3 billing guard stopped the
pod (storage kept, decision needed) | 4 evidence transfer failed (pod stopped, storage kept) | 5 bench failed (logs
saved, pod deleted) | 1 refused / launch failed with verified cleanup.

Every external command (runpodctl, ssh, scp) runs with a timeout; ssh/scp use BatchMode.  Paths and binaries come from
arguments/environment so the whole flow is tested with fakes (tests/test_cb2_l1b_bench_ctl.py), never against the cloud.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NAME = "cb2-l1b-bench-claude"
FILES = ["cb2_l1b_code.bundle", "cb2_l1b_code.rev", "cb2_l1b_pod.sh", "cb2_l1b_SHA256SUMS"]
INPUTS = {"tile_decoder_v1.pt": "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999",
          "cb2_kit_r2.tgz": "2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec",
          "s2bres/s2b_min.tgz": "81e31395144364bcc55b4f57ac97cb8cdc94947e5c22730a4e584d6b8603c5f8"}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "StrictHostKeyChecking=accept-new"]


class Ctl:
    def __init__(self, a):
        self.a = a
        self.rp = os.environ.get("RP_BIN", a.runpodctl)
        self.ssh_bin = os.environ.get("SSH_BIN", "ssh")
        self.scp_bin = os.environ.get("SCP_BIN", "scp")
        self.l1b = Path(a.exchange) / "l1b"
        self.status = self.l1b / "bench_status.txt"
        self.claim = self.l1b / "bench_claim.json"

    # ------------------------------------------------------------------ plumbing
    def log(self, msg: str):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with open(self.status, "a") as f:
            f.write(line + "\n")

    def run(self, cmd: list, timeout: float) -> tuple[int, str, str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s: {cmd[0]}"

    def rpj(self, args: list, timeout: float = 120):
        """runpodctl call -> (exit code, parsed json or None, stderr)."""
        code, out, err = self.run([self.rp, *args], timeout)
        try:
            j = json.loads(out) if out.strip() else None
        except json.JSONDecodeError:
            j = None
        return code, j, err

    def pod(self, pid: str):
        """Live pod record, or 'gone' when the API says not_found, or None when the lookup itself failed."""
        code, j, err = self.rpj(["pod", "get", pid], 60)
        if code != 0 and "not_found" in err:
            return "gone"
        if code != 0 or not isinstance(j, dict) or j.get("error"):
            return None
        return j

    def ssh(self, p: dict, remote: str, timeout: float):
        return self.run([self.ssh_bin, "-i", self.a.key, "-p", str(p["port"]), *SSH_OPTS, f"root@{p['ip']}", remote], timeout)

    def scp_up(self, p: dict, files: list, timeout: float):
        return self.run([self.scp_bin, "-i", self.a.key, "-P", str(p["port"]), *SSH_OPTS, *map(str, files),
                         f"root@{p['ip']}:/workspace/"], timeout)

    def scp_down(self, p: dict, remote: str, local: Path, timeout: float):
        return self.run([self.scp_bin, "-i", self.a.key, "-P", str(p["port"]), *SSH_OPTS, f"root@{p['ip']}:{remote}",
                         str(local)], timeout)

    @staticmethod
    def ssh_of(j: dict):
        ip = port = None
        s = j.get("ssh") or {}
        ip, port = s.get("ip") or s.get("host"), s.get("port")
        if not (ip and port):
            for q in ((j.get("runtime") or {}).get("ports") or []):
                if q.get("privatePort") == 22 and q.get("isIpPublic"):
                    ip, port = q.get("ip"), q.get("publicPort")
        return {"ip": ip, "port": port} if ip and port else None

    # ------------------------------------------------------------------ cleanup that is verified, never assumed
    def delete_verified(self, pid: str) -> bool:
        for attempt in range(1, self.a.retries + 1):
            code, _, err = self.rpj(["pod", "delete", pid], 120)
            if code != 0:
                self.log(f"delete {pid} attempt {attempt}: API error {err.strip()[:200]}")
            for _ in range(self.a.verify_polls):
                if self.pod(pid) == "gone":
                    self.log(f"pod {pid} deleted (verified: not_found)")
                    if self.claim.exists() and json.loads(self.claim.read_text()).get("id") == pid:
                        self.claim.rename(self.l1b / f"bench_claim_{pid}.closed.json")   # claim closed, kept as record
                    return True
                time.sleep(self.a.verify_wait)
        self.log(f"URGENT: pod {pid} could NOT be verified deleted after {self.a.retries} attempts -- it may still be "
                 f"billing. Delete it by hand: runpodctl pod delete {pid}")
        return False

    def stop_verified(self, pid: str) -> bool:
        for attempt in range(1, self.a.retries + 1):
            code, _, err = self.rpj(["pod", "stop", pid], 120)
            if code != 0:
                self.log(f"stop {pid} attempt {attempt}: API error {err.strip()[:200]}")
            for _ in range(self.a.verify_polls):
                j = self.pod(pid)
                if j == "gone" or (isinstance(j, dict) and j.get("desiredStatus") == "EXITED"):
                    self.log(f"pod {pid} stopped (verified: {'gone' if j == 'gone' else 'EXITED'}); its volume is kept")
                    return True
                time.sleep(self.a.verify_wait)
        self.log(f"URGENT: pod {pid} could NOT be verified stopped -- it may still be billing GPU time. "
                 f"Stop it by hand: runpodctl pod stop {pid}")
        return False

    # ------------------------------------------------------------------ identity
    def identity_ok(self, pid: str, j) -> bool:
        claim = json.loads(self.claim.read_text()) if self.claim.exists() else {}
        return isinstance(j, dict) and j.get("name") == NAME and claim.get("id") == pid and claim.get("name") == NAME

    def ours_on_account(self) -> list:
        code, j, err = self.rpj(["pod", "list", "--all"], 120)
        if code != 0 or not isinstance(j, list):
            raise SystemExit(f"cannot list pods ({err.strip()[:200]}): refusing to create anything blind")
        return [p for p in j if p.get("name") == NAME]

    # ------------------------------------------------------------------ launch
    def verify_package(self):
        bad = []
        for f in FILES:
            if not (self.l1b / f).exists():
                bad.append(f"missing {f}")
        sums = (self.l1b / "cb2_l1b_SHA256SUMS").read_text().split("\n") if (self.l1b / "cb2_l1b_SHA256SUMS").exists() else []
        for line in filter(None, sums):
            h, f = line.split()
            if hashlib.sha256((self.l1b / f).read_bytes()).hexdigest() != h:
                bad.append(f"checksum {f}")
        if not self.a.skip_input_sums:
            for f, h in INPUTS.items():
                p = Path(self.a.exchange) / f
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
                    bad.append(f"input {f}")
        if bad:
            raise SystemExit("package not verified: " + ", ".join(bad))

    def launch(self) -> int:
        self.verify_package()
        mine = self.ours_on_account()
        if self.a.pod_id:                                     # adopt an existing pod: only OUR claimed one
            j = self.pod(self.a.pod_id)
            if not self.identity_ok(self.a.pod_id, j):
                self.log(f"REFUSED: pod {self.a.pod_id} is not {NAME} with a matching claim -- not touching it")
                return 1
            pid = self.a.pod_id
        else:
            if mine:
                self.log(f"REFUSED: {NAME} already exists ({[p.get('id') for p in mine]}): one claim at a time")
                return 1
            if self.claim.exists():
                old = json.loads(self.claim.read_text())
                self.log(f"REFUSED: an earlier claim is still recorded ({old.get('id')}); resolve it first")
                return 1
            args = ["pod", "create", "--name", NAME, "--gpu-id", self.a.gpu, "--gpu-count", "1", "--cloud-type", self.a.cloud,
                    "--image", self.a.image, "--container-disk-in-gb", "40", "--volume-in-gb", "40", "--ports", "22/tcp", "--wait"]
            if self.a.cloud == "COMMUNITY":
                args.insert(-1, "--public-ip")
            code, j, err = self.rpj(args, self.a.create_timeout)
            pid = (j or {}).get("id") if isinstance(j, dict) else None
            if not pid:                                       # never assume nothing was created: reconcile by name
                found = self.ours_on_account()
                if len(found) == 1:
                    pid = found[0]["id"]
                    self.log(f"create output unusable (exit {code}); reconciled by name: {pid}")
                elif not found:
                    self.log(f"create failed (exit {code}: {err.strip()[:200]}); account shows no {NAME}: nothing to clean")
                    return 1
                else:
                    self.log(f"URGENT: create unclear and {len(found)} pods named {NAME} exist: {[p['id'] for p in found]}")
                    return 2
            self.claim.write_text(json.dumps({"id": pid, "name": NAME, "created": time.time()}) + "\n")
            self.log(f"created {pid} ({NAME}, 1x {self.a.gpu}, {self.a.cloud}); claim recorded")
        try:                                                  # until the watcher owns it, any failure deletes the pod
            p = None
            for _ in range(self.a.ssh_polls):
                j = self.pod(pid)
                p = self.ssh_of(j) if isinstance(j, dict) else None
                if p and self.ssh(p, "true", 60)[0] == 0:
                    break
                p = None
                time.sleep(self.a.ssh_wait)
            if not p:
                raise RuntimeError("ssh never became reachable")
            ups = [Path(self.a.exchange) / f for f in INPUTS] + [self.l1b / f for f in FILES]
            for attempt in range(1, 3):
                code, _, err = self.scp_up(p, ups, self.a.upload_timeout)
                if code == 0:
                    break
                self.log(f"upload attempt {attempt} failed: {err.strip()[:200]}")
            else:
                raise RuntimeError("upload failed")
            w = subprocess.Popen([sys.executable, __file__, "watch", "--pod-id", pid, *self.passthrough()],
                                 stdout=open(self.l1b / "bench_watcher.out", "a"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
            self.log(f"watcher started (pid {w.pid}); it owns cleanup of {pid} from now on")
            code, _, err = self.ssh(p, "cd /workspace && (PHASE=bench setsid nohup bash /workspace/cb2_l1b_pod.sh "
                                       "> /workspace/l1b-bench.log 2>&1 < /dev/null &); sleep 2; head -3 /workspace/l1b-bench.log", 60)
            self.log(f"bench start: exit {code}; {err.strip()[:120]}")
            return 0
        except Exception as e:
            self.log(f"launch failed before hand-off ({e}); deleting {pid}")
            return 1 if self.delete_verified(pid) else 2

    def passthrough(self) -> list:
        a = self.a
        return ["--exchange", a.exchange, "--key", a.key, "--runpodctl", a.runpodctl, "--maxh", str(a.maxh),
                "--poll", str(a.poll), "--grace", str(a.grace), "--retries", str(a.retries),
                "--verify-polls", str(a.verify_polls), "--verify-wait", str(a.verify_wait)]

    # ------------------------------------------------------------------ watch
    def watch(self) -> int:
        pid, t0 = self.a.pod_id, time.time()
        log_f, tgz = self.l1b / "bench_pod.log", self.l1b / "cb2_l1b_bench.tgz"
        while True:
            j = self.pod(pid)
            if j == "gone":
                self.log(f"pod {pid} no longer exists (deleted elsewhere?) -- watcher ends")
                return 0
            if isinstance(j, dict) and not self.identity_ok(pid, j):
                self.log(f"URGENT: pod {pid} is not our claimed {NAME} -- touching nothing")
                return 2
            if time.time() - t0 > self.a.maxh * 3600:         # billing guard BEFORE any slow transfer
                self.log(f"billing guard: past {self.a.maxh} h -- stopping {pid} (volume kept, decision needed)")
                return 3 if self.stop_verified(pid) else 2
            p = self.ssh_of(j) if isinstance(j, dict) else None
            if p:
                code, out, _ = self.ssh(p, "ps -eo cmd | grep -c '[b]ash /workspace/cb2_l1b_pod.sh'; "
                                           "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", 60)
                running = code == 0 and out.strip().split("\n")[0].strip() not in ("", "0")
                self.scp_down(p, "/workspace/l1b-bench.log", log_f, 120)
                text = log_f.read_text() if log_f.exists() else ""
                if "l1b BENCH finished" in text:
                    ok = False
                    for attempt in range(1, 3):
                        c1, _, e1 = self.scp_down(p, "/workspace/cb2_l1b_bench.tgz", tgz, self.a.download_timeout)
                        c2, rs, _ = self.ssh(p, "sha256sum /workspace/cb2_l1b_bench.tgz", 300)
                        if c1 == 0 and c2 == 0 and tgz.exists() and rs.split() and \
                                rs.split()[0] == hashlib.sha256(tgz.read_bytes()).hexdigest():
                            ok = True
                            break
                        self.log(f"evidence transfer attempt {attempt} failed ({e1.strip()[:120]})")
                    if not ok:
                        self.log("evidence NOT verified: stopping the pod with its volume kept, not deleting")
                        return 4 if self.stop_verified(pid) else 2
                    self.log(f"cb2_l1b_bench.tgz downloaded, sha256 MATCH {hashlib.sha256(tgz.read_bytes()).hexdigest()}")
                    for f in ("l1b-bench-sleep.json", "l1b-bench-lives-s2b.json", "l1b-bench-lives-learner.json"):
                        self.scp_down(p, f"/workspace/{f}", self.l1b / f, 120)
                    return 0 if self.delete_verified(pid) else 2
                if not running and time.time() - t0 > self.a.grace:
                    if not log_f.exists():
                        self.log("bench not running and its log could not be fetched: stopping, volume kept")
                        return 4 if self.stop_verified(pid) else 2
                    self.log("BENCH FAILED or stopped (see bench_pod.log): logs saved; deleting the pod")
                    return 5 if self.delete_verified(pid) else 2
            time.sleep(self.a.poll)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["launch", "watch"])
    ap.add_argument("--exchange", default="/mnt/c/Users/davei/ContinuousBrain/cb_exchange")
    ap.add_argument("--key", default="/home/davei/.runpod/ssh/runpodctl-ssh-key")
    ap.add_argument("--runpodctl", default=str(Path.home() / "projects/continuous_brain_explore/runpodctl"))
    ap.add_argument("--pod-id", default=None, help="launch: adopt this pod (only if it is our claimed one); watch: required")
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    ap.add_argument("--image", default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
    ap.add_argument("--maxh", type=float, default=3.0)
    ap.add_argument("--poll", type=float, default=120)
    ap.add_argument("--grace", type=float, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--verify-polls", type=int, default=12)
    ap.add_argument("--verify-wait", type=float, default=10)
    ap.add_argument("--ssh-polls", type=int, default=60)
    ap.add_argument("--ssh-wait", type=float, default=10)
    ap.add_argument("--create-timeout", type=float, default=1200)
    ap.add_argument("--upload-timeout", type=float, default=3600)
    ap.add_argument("--download-timeout", type=float, default=3600)
    ap.add_argument("--skip-input-sums", action="store_true", help="tests only")
    a = ap.parse_args()
    c = Ctl(a)
    if a.cmd == "watch" and not a.pod_id:
        raise SystemExit("watch needs --pod-id")
    return c.launch() if a.cmd == "launch" else c.watch()


if __name__ == "__main__":
    raise SystemExit(main())
