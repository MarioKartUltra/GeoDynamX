# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Qt objects in cyclic garbage: what the collector would free after a test, and the cycle each one
sits in. Switched on by ``DYNAMIX_GC_AUDIT=1`` (the Qt-test fixture in ``conftest.py``); each test's
findings are appended as JSON lines to ``DYNAMIX_GC_AUDIT_OUT`` (default ``gc_audit.jsonl`` in the
temporary directory), and ``python -m tests.gc_audit <file>`` groups them by cycle shape.

Under ``gc.DEBUG_SAVEALL`` a collection puts everything it would have freed into ``gc.garbage``
instead. The whole cycle is unreachable, so a chain back to a root is empty; the cycle is walked
through the referents inside the garbage itself, from the Qt wrapper back to it.
"""
from __future__ import annotations

import collections
import gc
import json
import os
import sys
import tempfile
import types



def out_path() -> str:
    return os.environ.get("DYNAMIX_GC_AUDIT_OUT") or os.path.join(tempfile.gettempdir(),
                                                                  "gc_audit.jsonl")


def _label(src, dst) -> str:
    """How ``src`` refers to ``dst``, as an attribute, key or index where that can be recovered."""
    if isinstance(src, dict):
        for k, v in src.items():
            if v is dst:
                return f"[{k!r:.40}]"
        return "[key]"
    if isinstance(src, (list, tuple)):
        for i, v in enumerate(src):
            if v is dst:
                return f"[{i}]"
    if isinstance(src, types.CellType):
        return ".cell_contents"
    if isinstance(src, types.MethodType):
        return ".__self__" if src.__self__ is dst else ".__func__"
    if isinstance(src, types.FunctionType):
        for name in ("__closure__", "__defaults__", "__kwdefaults__", "__dict__", "__globals__"):
            if getattr(src, name, None) is dst:
                return f".{name}"
    if getattr(src, "__dict__", None) is dst:
        return ".__dict__"
    for name in ("func", "args", "keywords"):                      # functools.partial
        if getattr(src, name, None) is dst:
            return f".{name}"
    return ""


def _name(o) -> str:
    t = type(o)
    if isinstance(o, types.FunctionType):
        return f"function {o.__qualname__}"
    if isinstance(o, types.MethodType):
        return f"method {getattr(o.__func__, '__qualname__', '?')}"
    return t.__qualname__ if t.__module__ in ("builtins", None) else f"{t.__module__}.{t.__qualname__}"


def _sccs(adj: dict[int, list[int]]) -> list[list[int]]:
    """Tarjan's strongly connected components of the garbage graph, iteratively: the cycles."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    out: list[list[int]] = []
    counter = 0
    for root in adj:
        if root in index:
            continue
        work = [(root, iter(adj[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(adj[w])))
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    if len(comp) > 1 or v in adj[v]:
                        out.append(comp)
    return out


def _cycle(start: int, members: set[int], adj: dict[int, list[int]],
           objs: dict[int, object]) -> list[str]:
    """The shortest path from ``start`` back to it inside its component, as names joined by the
    references between them."""
    parent: dict[int, int] = {}
    frontier = [start]
    seen = {start}
    while frontier:
        nxt = []
        for v in frontier:
            for w in adj[v]:
                if w == start:
                    path = [v]
                    while path[0] != start:
                        path.insert(0, parent[path[0]])
                    path.append(start)
                    out = [_name(objs[start])]
                    for a, b in zip(path, path[1:]):
                        out.append(f"-{_label(objs[a], objs[b])}-> {_name(objs[b])}")
                    return out
                if w in members and w not in seen:
                    seen.add(w)
                    parent[w] = v
                    nxt.append(w)
        frontier = nxt
    return []


def _own(name: str) -> bool:
    return "dynamix." in name or "tests." in name


def audit(nodeid: str) -> int:
    """Collect with SAVEALL, record each cycle that holds Qt wrappers (in it, or hanging off it) with
    one example path, free the garbage for real; returns how many Qt wrappers were found."""
    import shiboken6

    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        garbage = list(gc.garbage)
        gc.garbage.clear()
    finally:
        gc.set_debug(0)
    objs = {id(o): o for o in garbage}
    adj = {i: [id(r) for r in gc.get_referents(o) if id(r) in objs] for i, o in objs.items()}
    qt = {i for i, o in objs.items() if isinstance(o, shiboken6.Shiboken.Object)}
    found = []
    for comp in _sccs(adj):
        members = set(comp)
        reach, todo = set(members), list(members)
        while todo:
            for w in adj[todo.pop()]:
                if w not in reach:
                    reach.add(w)
                    todo.append(w)
        held = reach & qt
        if not held:
            continue
        names = {i: _name(objs[i]) for i in comp}
        start = min(comp, key=lambda i: (not _own(names[i]), i not in qt, names[i]))
        types_held = collections.Counter(_name(objs[i]) for i in held)
        found.append({"test": nodeid, "size": len(comp), "qt_in": len(members & qt),
                      "qt_held": len(held), "owner": names[start] if _own(names[start]) else "",
                      "qt_types": dict(types_held.most_common(6)),
                      "cycle": _cycle(start, members, adj, objs)})
    n = len(qt)
    garbage = objs = None
    gc.collect()
    if found:
        with open(out_path(), "a", encoding="utf-8") as f:
            for rec in found:
                f.write(json.dumps(rec) + "\n")
    return n


def report(path: str) -> str:
    """Markdown: the cycles' owners and shapes ranked by how many Qt wrappers they hold, each shape
    with its first test and one example path."""
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    tests = {r["test"] for r in recs}
    by_owner: dict[str, list[dict]] = collections.defaultdict(list)
    by_shape: dict[str, list[dict]] = collections.defaultdict(list)
    for r in recs:
        by_owner[r["owner"] or "(no app class on the cycle)"].append(r)
        by_shape[" ".join(r["cycle"])].append(r)
    held = sum(r["qt_held"] for r in recs)
    lines = [f"{len(recs)} cycles holding {held} Qt wrappers, after {len(tests)} tests; "
             f"{len(by_shape)} cycle shapes.", "", "## By owner (cycles, Qt wrappers held)", ""]
    for owner, rs in sorted(by_owner.items(), key=lambda kv: -sum(r["qt_held"] for r in kv[1])):
        lines.append(f"- `{owner}`: {len(rs)} cycles, {sum(r['qt_held'] for r in rs)} wrappers, "
                     f"in {len({r['test'] for r in rs})} tests")
    lines += ["", "## By cycle shape", ""]
    for shape, rs in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
        r0 = rs[0]
        lines.append(f"- **{len(rs)}** cycles, {sum(r['qt_held'] for r in rs)} wrappers held "
                     f"({r0['qt_in']} of {r0['qt_held']} on the cycle in the first), first in "
                     f"`{r0['test']}`; held: {r0['qt_types']}")
        lines.append(f"  `{shape}`")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else out_path()))
