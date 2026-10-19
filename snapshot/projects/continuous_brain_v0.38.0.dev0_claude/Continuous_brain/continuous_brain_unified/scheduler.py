"""Per-tick DAG scheduler. The runtime owns the clock (Charter: language is never the clock).

Within a tick, a component that reads channel c (non-delayed) runs after c's
writer. Delayed reads see the start-of-tick value and add no edge, which is
how recurrence is expressed. A same-tick cycle is a configuration error.
Ties are broken by slot order and then component name, so the order is a
pure function of the bound set.
"""

from __future__ import annotations

import heapq

from .components import Component
from .slots import REQUIRED_SLOTS


class SchedulingError(RuntimeError):
    pass


_SLOT_RANK = {s.name: i for i, s in enumerate(REQUIRED_SLOTS)}


def schedule(components: list[Component], external_channels: set[str]) -> list[Component]:
    writer: dict[str, Component] = {}
    for c in components:
        for d in c.writes:
            if d.name in writer:
                raise SchedulingError(f"{d.name!r} written by both {writer[d.name].name} and {c.name}")
            writer[d.name] = c
    for c in components:
        for ch in (*c.reads, *c.reads_delayed):
            if ch not in writer and ch not in external_channels:
                raise SchedulingError(f"{c.name} reads {ch!r}, which nothing writes (its slot may be unbound)")
    key = {c.name: (_SLOT_RANK.get(c.slot, len(_SLOT_RANK)), c.name) for c in components}
    succ: dict[str, set[str]] = {c.name: set() for c in components}
    indeg = {c.name: 0 for c in components}
    for c in components:
        for ch in c.reads:
            w = writer.get(ch)
            if w is not None and w.name != c.name and c.name not in succ[w.name]:
                succ[w.name].add(c.name)
                indeg[c.name] += 1
            elif w is not None and w.name == c.name:
                raise SchedulingError(f"{c.name} reads its own output {ch!r} in the same tick; declare it delayed")
    by_name = {c.name: c for c in components}
    ready = [key[n] for n, d in indeg.items() if d == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, name = heapq.heappop(ready)
        order.append(by_name[name])
        for s in sorted(succ[name]):
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready, key[s])
    if len(order) != len(components):
        stuck = sorted(n for n, d in indeg.items() if d > 0)
        raise SchedulingError(f"same-tick dependency cycle among {stuck}; make one edge delayed")
    return order
