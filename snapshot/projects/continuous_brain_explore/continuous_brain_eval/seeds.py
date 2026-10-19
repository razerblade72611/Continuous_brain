"""Seed-range registry (DAV-76 seed discipline).

`registry/seed_ranges.json` records every range the program has used or
reserved, including the legacy line's sealed TEST windows. A new family is
accepted only if it overlaps nothing registered. Spent sealed TEST ranges are
never reused; the registry refuses them by construction.

Known historical anomaly, kept visible: v0.33.5's TEST range (20000-20127)
lies inside v0.31.1's (20000-21023) in the historical record. It is flagged
in the registry and reported by `anomalies()`; it is not silently fixed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from continuous_brain_protocols import SeedFamily, SeedPurpose
from continuous_brain_unified.prereg import registry_dir


class SeedOverlapError(ValueError):
    pass


@dataclass(frozen=True)
class RegisteredRange:
    family: SeedFamily
    source: str
    note: str
    allowed_overlap_with: tuple[str, ...] = ()
    reserve: bool = False


def _key(f: SeedFamily) -> tuple:
    return (f.name, f.start, f.stop, f.purpose)


class SeedRegistry:
    def __init__(self, ranges: list[RegisteredRange], local_path: Path | None = None):
        self.ranges = ranges
        self.local_path = local_path

    @classmethod
    def load(cls, directory: str | os.PathLike | None = None) -> "SeedRegistry":
        d = registry_dir(directory)
        raw = json.loads((d / "seed_ranges.json").read_text())
        ranges = [cls._parse(r) for r in raw["ranges"]]
        local = d / "seed_ranges.local.jsonl"
        if local.exists():
            for line in local.read_text().splitlines():
                if not line.strip():
                    continue
                rr = cls._parse(json.loads(line))
                # a later ledger line for the same (name, range) supersedes an earlier one (e.g. "spent")
                ranges = [r for r in ranges if _key(r.family) != _key(rr.family)] + [rr]
        return cls(ranges, local)

    @staticmethod
    def _parse(r: dict) -> RegisteredRange:
        fam = SeedFamily(r["name"], int(r["start"]), int(r["stop"]), SeedPurpose(r["purpose"]),
                         bool(r.get("sealed", False)), bool(r.get("spent", False)))
        return RegisteredRange(fam, r.get("source", ""), r.get("note", ""),
                               tuple(r.get("allowed_overlap_with", ())), bool(r.get("reserve", False)))

    def overlaps(self, family: SeedFamily) -> list[RegisteredRange]:
        """Every registered range sharing a seed with `family`, compared by range (names are irrelevant)."""
        return [r for r in self.ranges if r.family.overlaps(family)]

    @staticmethod
    def _contains(parent: SeedFamily, child: SeedFamily) -> bool:
        return parent.start <= child.start and child.stop <= parent.stop

    @classmethod
    def _tolerated(cls, new: SeedFamily, old: RegisteredRange) -> bool:
        """Only explicit overlap relationships are tolerated.

        ENGINEERING is not a blanket exemption: independent engineering families
        must remain disjoint. A child may sit inside a registered engineering
        reserve (for example v0.38-ENGINEERING), and historical overlaps are
        tolerated only when named in allowed_overlap_with.
        """
        if new.name in old.allowed_overlap_with or old.family.name in getattr(new, "allowed_overlap_with", ()):
            return True
        return (old.reserve and new.purpose == SeedPurpose.ENGINEERING
                and old.family.purpose == SeedPurpose.ENGINEERING
                and cls._contains(old.family, new))

    def check_fresh(self, family: SeedFamily) -> None:
        """A never-registered family: refuse any overlap, except ENGINEERING inside an ENGINEERING reserve."""
        hits = [h for h in self.overlaps(family) if not self._tolerated(family, h)]
        if hits:
            desc = ", ".join(f"{h.family.name} [{h.family.start}-{h.family.stop}]" for h in hits)
            raise SeedOverlapError(f"{family.name} [{family.start}-{family.stop}] overlaps {desc}")

    def check_usable(self, family: SeedFamily) -> None:
        """A family about to be run: either fresh, or exactly a registered family that is not spent.

        A registered TEST family is usable once; `spend` marks it afterwards."""
        exact = [r for r in self.ranges if _key(r.family) == _key(family)]
        if not exact:
            self.check_fresh(family)
            return
        if any(r.family.spent for r in exact):
            raise SeedOverlapError(f"{family.name} [{family.start}-{family.stop}] is spent")
        if family.purpose == SeedPurpose.HISTORICAL:
            raise SeedOverlapError(f"{family.name} is a legacy-line range; this program never runs on it")
        others = [h for h in self.overlaps(family) if _key(h.family) != _key(family)
                  and not self._tolerated(family, h)]
        if others:
            raise SeedOverlapError(f"{family.name} overlaps {[h.family.name for h in others]}")

    def check_seeds(self, seeds, purpose_allowed=(SeedPurpose.ENGINEERING, SeedPurpose.DEV)) -> None:
        """Every seed must lie in a registered family of an allowed purpose, and in no sealed or spent family."""
        for s in seeds:
            hits = [r.family for r in self.ranges if r.family.start <= s <= r.family.stop]
            if not any(f.purpose in purpose_allowed for f in hits):
                raise SeedOverlapError(f"seed {s} is not inside a registered {[p.value for p in purpose_allowed]} family")
            bad = [f.name for f in hits if f.sealed or f.spent]
            if bad:
                raise SeedOverlapError(f"seed {s} lies in sealed/spent families {bad}")

    def _append(self, entry: dict) -> None:
        if self.local_path is None:
            raise RuntimeError("registry has no local ledger path")
        with open(self.local_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        # the directory entry too (strict), so a ledger created by this append survives a crash
        dfd = os.open(str(Path(self.local_path).parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def register(self, family: SeedFamily, source: str, note: str = "") -> None:
        """Append a new family to the local ledger after checking it is fresh."""
        self.check_fresh(family)
        if any(r.family.name == family.name for r in self.ranges):
            raise SeedOverlapError(f"family name {family.name!r} already registered")
        self._append({**family.as_dict(), "source": source, "note": note,
                      "registered_utc": datetime.now(timezone.utc).isoformat()})
        self.ranges.append(RegisteredRange(family, source, note, (), False))

    def spend(self, family: SeedFamily, source: str) -> None:
        """Record that a family has been opened (TEST families can then never run again)."""
        spent = SeedFamily(family.name, family.start, family.stop, family.purpose, family.sealed, True)
        self._append({**spent.as_dict(), "source": source, "note": "spent",
                      "registered_utc": datetime.now(timezone.utc).isoformat()})
        old = next((r for r in self.ranges if _key(r.family) == _key(family)), None)
        reserve = False if old is None else old.reserve
        allowed = () if old is None else old.allowed_overlap_with
        self.ranges = [r for r in self.ranges if _key(r.family) != _key(family)] + \
                      [RegisteredRange(spent, source, "spent", allowed, reserve)]

    def anomalies(self) -> list[str]:
        out = []
        for i, a in enumerate(self.ranges):
            for b in self.ranges[i + 1:]:
                if not a.family.overlaps(b.family):
                    continue
                known = b.family.name in a.allowed_overlap_with or a.family.name in b.allowed_overlap_with
                if known:
                    out.append(f"known overlap: {a.family.name} [{a.family.start}-{a.family.stop}] and "
                               f"{b.family.name} [{b.family.start}-{b.family.stop}]")
                    continue
                # Parent engineering reserves are bookkeeping containers, not collisions.
                reserve_overlap = ((a.reserve and self._contains(a.family, b.family) and
                                    a.family.purpose == b.family.purpose == SeedPurpose.ENGINEERING) or
                                   (b.reserve and self._contains(b.family, a.family) and
                                    a.family.purpose == b.family.purpose == SeedPurpose.ENGINEERING))
                if reserve_overlap:
                    continue
                out.append(f"UNEXPECTED overlap: {a.family.name} [{a.family.start}-{a.family.stop}] and "
                           f"{b.family.name} [{b.family.start}-{b.family.stop}]")
        return out

    def get(self, name: str) -> SeedFamily:
        for r in self.ranges:
            if r.family.name == name:
                return r.family
        raise KeyError(name)
