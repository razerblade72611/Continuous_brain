"""Fix for the r2 crash at the dream step (2026-10-08).

World model `_t` (and policy `_t`) called np.asarray(x) on every input. Imagination passes the policy's chosen
actions as CUDA tensors, which numpy cannot read, so `dream` crashed on the GPU (the CPU test passes, which is why
the suite did not catch it). The fix accepts torch tensors directly. Numpy inputs take exactly the same path as
before, so nothing already trained or measured changes. Run from the checkout root; idempotent.
"""
import sys
FIXES = {
    "continuous_brain_cb2/world_model.py": (
        "    def _t(self, x, dtype=torch.float32):\n        x = np.asarray(x)\n",
        "    def _t(self, x, dtype=torch.float32):\n        if isinstance(x, torch.Tensor):\n"
        "            return x.to(device=self.device, dtype=dtype)\n        x = np.asarray(x)\n"),
    "continuous_brain_cb2/policy.py": (
        "    def _t(self, x, dtype=torch.float32):\n        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)\n",
        "    def _t(self, x, dtype=torch.float32):\n        if isinstance(x, torch.Tensor):\n"
        "            return x.to(device=self.device, dtype=dtype)\n"
        "        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)\n"),
}
for path, (old, new) in FIXES.items():
    s = open(path).read()
    if new in s:
        print("already fixed:", path); continue
    if s.count(old) != 1:
        sys.exit(f"STOP: expected text not found exactly once in {path}; nothing changed there")
    open(path, "w").write(s.replace(old, new)); print("fixed:", path)
