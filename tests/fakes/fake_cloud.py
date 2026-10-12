#!/usr/bin/env python3
"""Fake runpodctl / ssh / scp for tests/test_cb2_l1b_bench_ctl.py (dispatch on the program name).  State lives in
$FAKE_STATE (json): pods, failure switches, and the fake pod's /workspace in $FAKE_REMOTE.  Never touches a network."""
import hashlib, json, os, shutil, sys, time
from pathlib import Path

S = Path(os.environ["FAKE_STATE"]); R = Path(os.environ["FAKE_REMOTE"])
st = json.loads(S.read_text())
prog = Path(sys.argv[0]).name
args = sys.argv[1:]


def save():
    S.write_text(json.dumps(st))


def err(code, msg):
    sys.stderr.write(json.dumps({"error": msg, "code": code}) + "\n")
    save(); sys.exit(1)


if prog == "runpodctl":
    st.setdefault("calls", []).append(args); pods = st.setdefault("pods", {})
    if args[:2] == ["pod", "list"]:
        print(json.dumps(list(pods.values())))
    elif args[:2] == ["pod", "get"]:
        p = pods.get(args[2])
        if not p: err("not_found", f"pod '{args[2]}' not found")
        print(json.dumps(p))
    elif args[:2] == ["pod", "create"]:
        name = args[args.index("--name") + 1]
        pid = st.get("next_id", "newpod00001")
        pods[pid] = {"id": pid, "name": name, "desiredStatus": "RUNNING", "runtimeStatus": "running",
                     "costPerHr": st.get("rate", 0.34)}
        if st.get("create_garbage"): print("not json")
        else: print(json.dumps(pods[pid]))
    elif args[:2] == ["ssh", "info"]:
        p = pods.get(args[2])
        if not p: err("not_found", f"pod '{args[2]}' not found")
        if p["desiredStatus"] != "RUNNING" or not st.get("ssh_ok", True):
            print(json.dumps({"error": "pod not ready: stopped", "id": args[2]}))
        else:
            print(json.dumps({"id": args[2], "name": p["name"], "ip": "10.0.0.1", "port": 2222, "ssh_command": "ssh ..."}))
    elif args[:2] == ["pod", "delete"]:
        if st.get("delete_fail", 0) > 0:
            st["delete_fail"] -= 1; err("api_error", "internal error")
        pods.pop(args[2], None); print(json.dumps({"deleted": args[2]}))
    elif args[:2] == ["pod", "stop"]:
        if args[2] in pods:
            pods[args[2]]["desiredStatus"] = "EXITED"
            if not st.get("stop_runtime_lags"): pods[args[2]]["runtimeStatus"] = "stopped"
        print(json.dumps({"stopped": args[2]}))
    save()
elif prog == "ssh":
    remote = args[-1]
    if remote == "true": sys.exit(0 if st.get("ssh_ok", True) else 255)
    if "grep -c" in remote: print("1" if st.get("running", True) else "0"); sys.exit(0)
    if remote.startswith("sha256sum"):
        f = R / remote.split()[-1].split("/")[-1]; print(hashlib.sha256(f.read_bytes()).hexdigest() + "  " + str(f)); sys.exit(0)
    st["bench_started"] = True; save(); sys.exit(0)
elif prog == "scp":
    src, dst = args[-2], args[-1]
    if dst.endswith(":/workspace/"):                      # upload
        if st.get("upload_fail"): sys.exit(1)
        sys.exit(0)
    name = src.split(":", 1)[1].split("/")[-1]
    if name.endswith(".tgz") and st.get("download_hang"): time.sleep(30)
    if name.endswith(".tgz") and st.get("download_fail"): sys.exit(1)
    if (R / name).exists(): shutil.copy(R / name, dst); sys.exit(0)
    sys.exit(1)
