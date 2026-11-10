"""FATE-ARC: FATE's bidirectional elimination (forward reachability ∩ backward preimage, certified) applied to ARC demo pairs.
Ops are EXACT bijections on grids with exact inverses (so the backward pass is exact, no brute-force preimage).
A program = forward ops s1 then (conceptually) s2; found by meet-in-the-middle, joined across ALL demo pairs.
Optional final global colour map (same for every pair) = 'pure' vs '+cmap' arms. Everything is reported, nothing dropped."""
from __future__ import annotations
import itertools, json, sys, time
import numpy as np

def _r90(g): return np.rot90(g, -1)      # clockwise
def _r270(g): return np.rot90(g, 1)
OPS, INV = {}, {}
def add(name, f, inv): OPS[name], INV[name] = f, inv
add("rot90", _r90, _r270); add("rot270", _r270, _r90)
add("rot180", lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 2))
add("flipH", lambda g: g[:, ::-1], lambda g: g[:, ::-1]); add("flipV", lambda g: g[::-1], lambda g: g[::-1])
add("transpose", lambda g: g.T, lambda g: g.T); add("antitranspose", lambda g: np.rot90(g, 2).T, lambda g: np.rot90(g, 2).T)
for k in (1, 2):
    for ax, nm in ((0, "V"), (1, "H")):
        add(f"roll{nm}+{k}", (lambda g, k=k, ax=ax: np.roll(g, k, ax)), (lambda g, k=k, ax=ax: np.roll(g, -k, ax)))
        add(f"roll{nm}-{k}", (lambda g, k=k, ax=ax: np.roll(g, -k, ax)), (lambda g, k=k, ax=ax: np.roll(g, k, ax)))
def _swap(a, b):
    def f(g):
        o = g.copy(); o[g == a] = b; o[g == b] = a; return o
    return f
for a in range(10):
    for b in range(a + 1, 10):
        add(f"swap{a}{b}", _swap(a, b), _swap(a, b))
NAMES = list(OPS)

def key(g): return (g.shape, g.tobytes())

def _enum(g, table, depth):
    """all (grid -> shortest op sequence) reachable in <= depth steps using op table"""
    seen = {key(g): ()}; frontier = [(g, ())]
    for _ in range(depth):
        nxt = []
        for h, seq in frontier:
            for n in NAMES:
                o = table[n](h); k = key(o)
                if k not in seen: seen[k] = seq + (n,); nxt.append((o, seq + (n,)))
        frontier = nxt
    return seen

def apply_seq(g, seq):
    for n in seq: g = OPS[n](g)
    return g

def solve(train, test_inputs, d_fwd=2, d_bwd=2, use_cmap=False, max_progs=50):
    """returns up to 2 predicted outputs per test input (list of lists)"""
    X = [np.array(p["input"]) for p in train]; Y = [np.array(p["output"]) for p in train]
    progs = []
    if all(x.shape == y.shape or x.shape == y.T.shape for x, y in zip(X, Y)):
        fwd0 = _enum(X[0], OPS, d_fwd); bwd0 = _enum(Y[0], INV, d_bwd)   # forward from X0, exact preimages from Y0
        # join on grid equality, i.e. forward seq s1 and backward seq s2 meet at the same grid
        for k, s1 in fwd0.items():
            if k in bwd0:
                s2 = bwd0[k]                      # sequence of INVERSE ops applied to Y0 to reach the meeting grid
                prog = s1 + tuple(reversed(s2))   # undo the inverses: apply originals in reverse order
                progs.append(prog)
        # certified elimination across the other pairs
        ok = []
        for p in progs:
            if all(np.array_equal(apply_seq(x, p), y) for x, y in zip(X[1:], Y[1:])): ok.append(p)
        progs = sorted(set(ok), key=len)
    cmaps = []
    if use_cmap and not progs:
        cands = []
        gens = list(itertools.chain([()], ((n,) for n in NAMES if not n.startswith("swap"))))
        for p in gens:
            try: Z = [apply_seq(x, p) for x in X]
            except Exception: continue
            if not all(z.shape == y.shape for z, y in zip(Z, Y)): continue
            m = {}; good = True
            for z, y in zip(Z, Y):
                for a, b in zip(z.ravel(), y.ravel()):
                    if m.setdefault(int(a), int(b)) != int(b): good = False; break
                if not good: break
            if good: cands.append((p, m))
        cmaps = cands[:max_progs]
    outs = []
    for t in test_inputs:
        t = np.array(t); preds = []
        for p in progs[:max_progs]:
            try: o = apply_seq(t, p)
            except Exception: continue
            if not any(np.array_equal(o, q) for q in preds): preds.append(o)
        for p, m in cmaps:
            o = apply_seq(t, p); o = np.vectorize(lambda v: m.get(int(v), int(v)))(o)
            if not any(np.array_equal(o, q) for q in preds): preds.append(o)
        outs.append(preds[:2])
    return outs, len(progs), len(cmaps)
