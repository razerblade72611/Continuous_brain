"""The experience archive: every transition the individual lived, practised or dreamed, at the scale of millions.

Layout (one directory per stream)::

    <root>/<stream>/stream.json                         where, who may write it, the individual it belongs to
    <root>/<stream>/shard-000000.json                   the shard's manifest (written last: no manifest, no shard)
    <root>/<stream>/shard-000000.student.parquet        what the individual could sense and what it did
    <root>/<stream>/shard-000000.teacher.parquet        what a teacher knew or would have done (never student input)
    <root>/<stream>/shard-000000.organ.parquet          what its own organs reported (memory readouts)

Provenance has two axes.  ``where`` is per stream: LIVED (the individual's one chronology), PRACTICE (a real
simulator, not lived), DREAMED (imagined by its world model).  ``who`` is per row: SELF (the individual chose the
action) or TEACHER (a teacher chose it).  A COUNTERFACTUAL label is a column of the teacher namespace: what the
teacher would have done when it did not act.  Only LIVED x SELF rows are things that happened to the individual
(``lived_rows``); an individual has exactly one LIVED stream, and the archive refuses a second.

Shards hold ``SHARD_ROWS`` transitions in row groups of ``ROW_GROUP`` (zstd).  Each manifest records the SHA-256 of
every namespace file; LIVED manifests also chain to the previous manifest's hash, so the lived record is
tamper-evident.  ``verify`` recomputes all of it.

Readers name the namespaces they open (``namespaces=("student",)``), so a student's loader never opens the
teacher's files.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FORMAT = "cb2-archive/1"
SCHEMA_VERSION = 1
SHARD_ROWS = 65536
ROW_GROUP = 4096
WHERE = ("LIVED", "PRACTICE", "DREAMED")
WHO = ("SELF", "TEACHER")
FRAME = (64, 64, 3)
FRAME_BYTES = int(np.prod(FRAME))
INVENTORY = ("health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond",
             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")
VIEW = 7 * 9
FEATURES = 62
CROP = 33                                      # remembered map, egocentric window (tiles)


def _fixed(t, n):
    return pa.list_(t, n)


SCHEMAS = {
    "student": pa.schema([
        ("tick", pa.int64()), ("life", pa.int32()), ("step", pa.int32()), ("seed", pa.int64()),
        ("frame", pa.binary(FRAME_BYTES)),
        ("inventory", _fixed(pa.int8(), len(INVENTORY))), ("pos", _fixed(pa.int16(), 2)),
        ("facing", _fixed(pa.int8(), 2)), ("sleeping", pa.bool_()),
        ("messages", pa.string()),
        ("action", pa.int8()), ("utterance", pa.string()), ("who", pa.int8()),
        ("reward", pa.float32()), ("reward_caretaker", pa.float32()), ("done", pa.bool_()), ("dying", pa.bool_()),
        ("store_hash", pa.string()),           # the individual's state hash after this tick ("" off the lived stream)
    ]),
    "teacher": pa.schema([
        ("tick", pa.int64()),
        ("teacher_action", pa.int8()),         # what the teacher did, or would have done (COUNTERFACTUAL)
        ("teacher_option", pa.int8()),
        ("view_material", _fixed(pa.int8(), VIEW)), ("view_object", _fixed(pa.int8(), VIEW)),
    ]),
    "organ": pa.schema([
        ("tick", pa.int64()),
        ("features", _fixed(pa.float16(), FEATURES)),   # belief features from its own memory (no routine state)
        ("map_crop", _fixed(pa.int8(), CROP * CROP)),   # its remembered map around it; -1 unknown
    ]),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(_canon(obj))
    os.replace(tmp, path)


class ArchiveError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- writing
class _Buffers:
    def __init__(self):
        self.cols: dict[str, dict[str, list]] = {ns: {f.name: [] for f in s} for ns, s in SCHEMAS.items()}
        self.n = {ns: 0 for ns in SCHEMAS}

    def add(self, ns: str, row: dict) -> None:
        cols = self.cols[ns]
        if set(row) != set(cols):
            missing, extra = set(cols) - set(row), set(row) - set(cols)
            raise ArchiveError(f"{ns} row: missing {sorted(missing)}, unknown {sorted(extra)}")
        for k, v in row.items():
            cols[k].append(v)
        self.n[ns] += 1

    def table(self, ns: str) -> pa.Table:
        return pa.Table.from_pydict(self.cols[ns], schema=SCHEMAS[ns])

    def clear(self, ns: str) -> None:
        for v in self.cols[ns].values():
            v.clear()
        self.n[ns] = 0


class StreamWriter:
    """Appends one stream's transitions.  ``append(student, teacher=None, organ=None)``; ``close()`` seals the
    last, partial shard.  A teacher or organ row, once given for a shard, must be given for every row of it."""

    def __init__(self, root, stream: str, where: str, individual: str, provenance: dict | None = None,
                 shard_rows: int = SHARD_ROWS, row_group: int = ROW_GROUP):
        if where not in WHERE:
            raise ArchiveError(f"where must be one of {WHERE}")
        if shard_rows % row_group:
            raise ArchiveError("shard_rows must be a multiple of row_group")
        self.root, self.stream, self.where, self.individual = Path(root), stream, where, individual
        self.dir = self.root / stream
        self.provenance = dict(provenance or {})
        self.shard_rows, self.row_group = shard_rows, row_group
        arch = Archive(self.root)
        if where == "LIVED":
            for s in arch.streams():
                if s["where"] == "LIVED" and s["individual"] == individual and s["stream"] != stream:
                    raise ArchiveError(f"{individual} already has a lived stream ({s['stream']}): one chronology")
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "stream.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if (m["where"], m["individual"]) != (where, individual):
                raise ArchiveError(f"stream {stream} exists as {m['where']} of {m['individual']}")
        else:
            _atomic_json(meta, {"format": FORMAT, "stream": stream, "where": where, "individual": individual})
        done = arch.manifests(stream)
        self.shard = len(done)
        self.prev = done[-1]["manifest_sha"] if done else None
        self.next_tick = (done[-1]["tick_range"][1] + 1) if done else 0
        self._open()

    # internal --------------------------------------------------------------------------------------------
    def _open(self):
        self.buf = _Buffers()
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.rows, self.ns_used = 0, None
        self.lives, self.seeds, self.first_tick = set(), set(), None
        self.who_counts = [0, 0]

    def _path(self, ns: str) -> Path:
        return self.dir / f"shard-{self.shard:06d}.{ns}.parquet"

    def _flush(self):
        for ns in self.ns_used:
            if not self.buf.n[ns]:
                continue
            if ns not in self.writers:
                self.writers[ns] = pq.ParquetWriter(str(self._path(ns)) + ".tmp", SCHEMAS[ns], compression="zstd",
                                                    compression_level=3, data_page_size=4 << 20)
            self.writers[ns].write_table(self.buf.table(ns), row_group_size=self.row_group)
            self.buf.clear(ns)

    def _seal(self):
        if not self.rows:
            return
        self._flush()
        files = {}
        for ns, w in self.writers.items():
            w.close()
            os.replace(str(self._path(ns)) + ".tmp", self._path(ns))
            files[ns] = {"path": self._path(ns).name, "sha256": sha256_file(self._path(ns))}
        man = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "stream": self.stream, "where": self.where,
               "individual": self.individual, "shard": self.shard, "rows": self.rows,
               "tick_range": [self.first_tick, self.first_tick + self.rows - 1],
               "lives": sorted(self.lives), "seeds": sorted(self.seeds),
               "who": {WHO[i]: c for i, c in enumerate(self.who_counts)}, "files": files,
               "provenance": self.provenance, "prev_manifest_sha": self.prev if self.where == "LIVED" else None}
        man["manifest_sha"] = hashlib.sha256(_canon(man)).hexdigest()
        _atomic_json(self.dir / f"shard-{self.shard:06d}.json", man)
        self.prev = man["manifest_sha"]
        self.shard += 1
        self._open()

    # public ----------------------------------------------------------------------------------------------
    def append(self, student: dict, teacher: dict | None = None, organ: dict | None = None) -> int:
        """Append one transition; returns its tick in this stream."""
        used = ("student",) + (("teacher",) if teacher is not None else ()) + (("organ",) if organ is not None else ())
        if self.ns_used is None:
            self.ns_used = used
        elif used != self.ns_used:
            raise ArchiveError(f"this shard carries namespaces {self.ns_used}, got {used}")
        tick = self.next_tick
        who = int(student["who"])
        if self.where == "DREAMED" and student.get("store_hash"):
            raise ArchiveError("a dream has no store hash: it did not happen to the individual")
        self.buf.add("student", {**student, "tick": tick})
        if teacher is not None:
            self.buf.add("teacher", {**teacher, "tick": tick})
        if organ is not None:
            self.buf.add("organ", {**organ, "tick": tick})
        if self.first_tick is None:
            self.first_tick = tick
        self.rows += 1
        self.next_tick += 1
        self.lives.add(int(student["life"]))
        self.seeds.add(int(student["seed"]))
        self.who_counts[who] += 1
        if self.buf.n["student"] >= self.row_group:
            self._flush()
        if self.rows >= self.shard_rows:
            self._seal()
        return tick

    def close(self) -> None:
        self._seal()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()


# ---------------------------------------------------------------------------------------------------- reading
@dataclass
class ShardRef:
    stream: str
    where: str
    manifest: dict
    dir: Path

    def path(self, ns: str) -> Path:
        if ns not in self.manifest["files"]:
            raise ArchiveError(f"shard {self.manifest['shard']} of {self.stream} has no {ns} namespace")
        return self.dir / self.manifest["files"][ns]["path"]


class Archive:
    def __init__(self, root):
        self.root = Path(root)

    def streams(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            meta = d / "stream.json"
            if meta.exists():
                out.append(json.loads(meta.read_text()))
        return out

    def manifests(self, stream: str) -> list[dict]:
        d = self.root / stream
        if not d.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(d.glob("shard-*.json"))]

    def shards(self, where: tuple[str, ...] = WHERE, individual: str | None = None) -> list[ShardRef]:
        out = []
        for s in self.streams():
            if s["where"] in where and (individual is None or s["individual"] == individual):
                out += [ShardRef(s["stream"], s["where"], m, self.root / s["stream"]) for m in self.manifests(s["stream"])]
        return out

    def rows(self, where=WHERE, individual=None) -> int:
        return sum(r.manifest["rows"] for r in self.shards(where, individual))

    def read(self, shard: ShardRef, namespaces=("student",), columns: dict | None = None) -> dict[str, pa.Table]:
        """The shard's tables, only for the namespaces named."""
        cols = columns or {}
        return {ns: pq.read_table(shard.path(ns), columns=cols.get(ns)) for ns in namespaces}

    def row_groups(self, shard: ShardRef, ns: str) -> int:
        return pq.ParquetFile(shard.path(ns)).num_row_groups

    def read_group(self, shard: ShardRef, ns: str, g: int, columns=None) -> pa.Table:
        return pq.ParquetFile(shard.path(ns)).read_row_group(g, columns=columns)

    def lived_rows(self, individual: str, columns=None):
        """What happened to the individual: LIVED x SELF rows of its one chronology, in order."""
        for ref in self.shards(("LIVED",), individual):
            t = pq.read_table(ref.path("student"), columns=None if columns is None else list(set(columns) | {"who"}))
            t = t.filter(pc.equal(t["who"], WHO.index("SELF")))
            yield t if columns is None else t.select(columns)

    def verify(self) -> dict:
        """Recompute every file hash, every manifest hash and the lived chains.  Raises on the first fault."""
        n = 0
        for s in self.streams():
            prev = None
            for m in self.manifests(s["stream"]):
                body = {k: v for k, v in m.items() if k != "manifest_sha"}
                if hashlib.sha256(_canon(body)).hexdigest() != m["manifest_sha"]:
                    raise ArchiveError(f"{s['stream']} shard {m['shard']}: manifest altered")
                for ns, f in m["files"].items():
                    if sha256_file(self.root / s["stream"] / f["path"]) != f["sha256"]:
                        raise ArchiveError(f"{s['stream']} shard {m['shard']}: {ns} file altered")
                if s["where"] == "LIVED" and m["prev_manifest_sha"] != prev:
                    raise ArchiveError(f"{s['stream']} shard {m['shard']}: lived chain broken")
                prev = m["manifest_sha"]
                n += 1
        return {"shards": n, "rows": self.rows()}

    def seeds_used(self) -> set[int]:
        """Every world seed any stream has touched: a seed is held out only if it is not here."""
        out = set()
        for s in self.streams():
            for m in self.manifests(s["stream"]):
                out |= set(m["seeds"])
        return out


# ---------------------------------------------------------------------------------------------------- decoding
def frames(table: pa.Table) -> np.ndarray:
    """uint8[N, 64, 64, 3] from a student table's frame column, without a Python loop."""
    col = table.column("frame").combine_chunks()
    return np.frombuffer(col.buffers()[1], np.uint8, count=len(col) * FRAME_BYTES,
                         offset=col.offset * FRAME_BYTES).reshape((len(col),) + FRAME)


def fixed(table: pa.Table, name: str, dtype) -> np.ndarray:
    col = table.column(name).combine_chunks()
    n = col.type.list_size
    vals = col.values.to_numpy(zero_copy_only=False)
    return np.asarray(vals, dtype).reshape(len(col), n)
