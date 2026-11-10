from __future__ import annotations
import itertools, json, math, sys, time
from typing import Callable, Iterable, Optional
import numpy as np

# ===== ledger.py =====
"""Admissibility ledger for grid tasks.

Mined from demonstration pairs. An invariant is active iff it holds on every demo.
A candidate output is admissible iff every active invariant holds. Decoding is a
projection of the model's per-cell distribution onto the admissible set."""
import numpy as np
from scipy.optimize import linear_sum_assignment

NC = 10

def counts(g): return np.bincount(g.ravel(), minlength=NC)

class Ledger:
    def __init__(self, demos):
        self.demos = demos
        ins = [d[0] for d in demos]; outs = [d[1] for d in demos]
        # shape rule: first consistent candidate in a fixed order
        self.shape_rule = None
        for name, f in (("same", lambda s: s), ("transpose", lambda s: (s[1], s[0]))):
            if all(f(i.shape) == o.shape for i, o in zip(ins, outs)):
                self.shape_rule = name; self._sf = f; break
        # palette: colours seen in outputs but absent from the matching input
        self.new_colors = set()
        for i, o in demos:
            self.new_colors |= set(np.unique(o)) - set(np.unique(i))
        # count conservation (exact) and its weaker sorted-profile form
        same_n = all(i.size == o.size for i, o in demos)
        self.count_conserved = same_n and all((counts(i) == counts(o)).all() for i, o in demos)
        self.profile_conserved = same_n and all(
            (np.sort(counts(i)) == np.sort(counts(o))).all() for i, o in demos)

    # ---- queries -------------------------------------------------------
    def out_shape(self, inp):
        return self._sf(inp.shape) if self.shape_rule else inp.shape

    def palette(self, inp):
        return sorted(set(np.unique(inp)) | self.new_colors)

    def palette_mask(self, inp):
        m = np.zeros(NC, bool); m[self.palette(inp)] = True; return m

    def admits(self, inp, out):
        if self.shape_rule and out.shape != self.out_shape(inp): return False
        if not self.palette_mask(inp)[np.unique(out)].all(): return False
        if self.count_conserved and not (counts(inp) == counts(out)).all(): return False
        if (not self.count_conserved) and self.profile_conserved:
            if not (np.sort(counts(inp)) == np.sort(counts(out))).all(): return False
        return True

    # ---- decoders (logp has shape [N, NC], rows in raster order) ----------------
    def decode_unconstrained(self, logp): return logp.argmax(1)

    def decode_mask(self, logp, inp):
        lp = np.where(self.palette_mask(inp)[None, :], logp, -1e9)
        return lp.argmax(1)

    def decode_sequential(self, logp, inp):
        """State-tracking decoder: the state is the remaining colour budget.
        Each step may only emit a colour with budget left; the budget shrinks by one.
        Without conservation it reduces to the palette mask."""
        lp = np.where(self.palette_mask(inp)[None, :], logp, -1e9)
        if not self.count_conserved: return lp.argmax(1)
        rem = counts(inp).copy(); out = np.empty(len(lp), int)
        for t in range(len(lp)):
            row = np.where(rem > 0, lp[t], -1e18)
            c = int(row.argmax()); out[t] = c; rem[c] -= 1
        return out

    def decode_projection(self, logp, inp):
        """Exact projection: best admissible assignment of cells to colour slots."""
        lp = np.where(self.palette_mask(inp)[None, :], logp, -1e9)
        if not self.count_conserved: return lp.argmax(1)
        slots = np.repeat(np.arange(NC), counts(inp))
        cost = -lp[:, slots]
        r, c = linear_sum_assignment(cost)
        out = np.empty(len(lp), int); out[r] = slots[c]; return out

# ===== fate_arc.py =====
"""FATE-ARC: FATE's bidirectional elimination (forward reachability ∩ backward preimage, certified) applied to ARC demo pairs.
Ops are EXACT bijections on grids with exact inverses (so the backward pass is exact, no brute-force preimage).
A program = forward ops s1 then (conceptually) s2; found by meet-in-the-middle, joined across ALL demo pairs.
Optional final global colour map (same for every pair) = 'pure' vs '+cmap' arms. Everything is reported, nothing dropped."""

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

# ===== select_pool.py =====
"""Selection layers over recorded candidate pools (see PREREG.md for the pre-registered definitions).

A pool is a list of samples {solution: ndarray, beam_score: float, score_aug: [8 NLLs]} for ONE test output.
Every function returns an ordered list of candidate grids (ndarray); the caller takes the first two.

Framework rule enforced here (FATE spec sec.2 + Admissibility paper): a learned/soft score may ORDER candidates, only a
declared admissibility invariant may DEMOTE one, and nothing is dropped from a slot unless >= 2 admissible candidates exist.
"""

import math
from typing import Callable, Optional
import numpy as np

LN10 = math.log(10.0)


def group(samples):
    G = {}
    for s in samples:
        G.setdefault(tuple(map(tuple, s["solution"])), []).append(s)
    return G


def kgmon(g):                      # the pipeline's own default (arc_decoder.getter_kgmon)
    return len(g) - float(np.mean([np.mean(s["score_aug"]) for s in g]))


def probmul3(g, baseline=3):       # the pipeline's alternative (arc_decoder.getter_full_probmul_3)
    inf = np.sum([baseline - s["beam_score"] for s in g])
    aug = np.mean([np.sum([baseline - x for x in s["score_aug"]]) for s in g])
    return float(inf + aug)


def _arr(h):
    return np.array(h, dtype=int)


def v0(samples, scorer=kgmon):
    G = group(samples)
    return [_arr(h) for h in sorted(G, key=lambda h: -scorer(G[h]))]


def v1_mask(samples, ledger: Ledger, test_input, scorer=kgmon):
    """SSM mask: admissible candidates first (V0 order inside each block); excluded ones only fill slots when fewer than 2 are admissible."""
    G = group(samples)
    order = sorted(G, key=lambda h: -scorer(G[h]))
    adm = [h for h in order if ledger.admits(test_input, _arr(h))]
    exc = [h for h in order if h not in set(adm)]
    return [_arr(h) for h in adm + exc], len(exc)


def fate_forced(fate_outs):
    """fate_outs: list of arrays certified by FATE (reproduce every demo pair). Returned as-is, de-duplicated."""
    out = []
    for o in fate_outs or []:
        if not any(o.shape == q.shape and (o == q).all() for q in out):
            out.append(np.array(o))
    return out


def merge_front(front, rest):
    out = []
    for g in list(front) + list(rest):
        if not any(g.shape == q.shape and (g == q).all() for q in out):
            out.append(g)
    return out


def v2_fate(samples, fate_outs, scorer=kgmon):
    return merge_front(fate_forced(fate_outs)[:1], v0(samples, scorer))


def v3_both(samples, ledger, test_input, fate_outs, scorer=kgmon):
    ranked, n_exc = v1_mask(samples, ledger, test_input, scorer)
    return merge_front(fate_forced(fate_outs)[:1], ranked)


def v4_control_soft_delete(samples, scorer=kgmon, delta=LN10):
    """CONTROL that violates the framework's rule: delete by a learned score."""
    G = group(samples)
    mean_aug = {h: float(np.mean([np.mean(s["score_aug"]) for s in G[h]])) for h in G}
    best = min(mean_aug.values())
    keep = [h for h in G if mean_aug[h] <= best + delta]
    return [_arr(h) for h in sorted(keep, key=lambda h: -scorer(G[h]))]


def top2_hit(ranked, truth) -> bool:
    t = np.array(truth)
    return any(g.shape == t.shape and (g == t).all() for g in ranked[:2])


def rank_of(ranked, truth) -> Optional[int]:
    t = np.array(truth)
    for i, g in enumerate(ranked):
        if g.shape == t.shape and (g == t).all():
            return i
    return None

# ===== entry.py =====
# ---- entry points used by the notebook's final cell (appended after the vendored modules) ----
import numpy as np

_FATE_CACHE = {}

def _fate_outputs(tid, task):
    if tid not in _FATE_CACHE:
        try:
            outs, nprog, ncm = solve(task["train"], [p["input"] for p in task["test"]], 2, 2, True)
            _FATE_CACHE[tid] = outs if (nprog or ncm) else None
        except Exception as e:
            print("FATE exact failed", tid, repr(e)[:80]); _FATE_CACHE[tid] = None
    return _FATE_CACHE[tid]

def select_all(decoded_results, challenges, variant="V1", scorer=None, verbose=False):
    """decoded_results: {task_idx: {subkey: sample}}, challenges: {task: json}. Returns {task_idx: [ndarray, ...]} like ArcDecoder.run_selection_algo.
    Any error on a task falls back to the pipeline's own baseline ordering for that task."""
    scorer = scorer or kgmon
    out, stats = {}, dict(tasks=0, excluded_candidates=0, tasks_mask_changed=0, fate_used=0, fallbacks=0)
    for bk, guesses in decoded_results.items():
        samples = list(guesses.values()); base = v0(samples, scorer)
        try:
            tid, idx = bk.split("_"); task = challenges[tid]; idx = int(idx)
            test_in = np.array(task["test"][idx]["input"]); demos = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
            ranked = base
            if variant in ("V1", "V3"):
                L = Ledger(demos); ranked, n_exc = v1_mask(samples, L, test_in, scorer)
                stats["excluded_candidates"] += n_exc
                if n_exc and not all(a.shape == b.shape and (a == b).all() for a, b in zip(ranked[:2], base[:2])): stats["tasks_mask_changed"] += 1
            if variant in ("V2", "V3"):
                F = _fate_outputs(tid, task)
                if F is not None and F[idx]:
                    ranked = merge_front(fate_forced(F[idx])[:1], ranked); stats["fate_used"] += 1
            out[bk] = ranked
        except Exception as e:
            stats["fallbacks"] += 1; out[bk] = base
            if verbose: print("select_all fallback", bk, repr(e)[:100])
        stats["tasks"] += 1
    return out, stats

def fallback_for_missing(challenges, covered):
    """CPU-only guesses for tasks the GPU budget never reached (the baseline leaves them at [[0]])."""
    out = {}
    for tid, task in challenges.items():
        for idx in range(len(task["test"])):
            bk = f"{tid}_{idx}"
            if bk in covered: continue
            ti = np.array(task["test"][idx]["input"]); guesses = []
            F = _fate_outputs(tid, task)
            if F is not None and F[idx]: guesses += list(fate_forced(F[idx]))
            try:
                demos = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
                if all(i.shape == o.shape for i, o in demos):
                    m = {}
                    for i, o in demos:
                        for a, b in zip(i.ravel(), o.ravel()): m.setdefault(int(a), {}).setdefault(int(b), 0); m[int(a)][int(b)] += 1
                    cm = np.vectorize(lambda v: max(m[int(v)], key=m[int(v)].get) if int(v) in m else int(v))(ti)
                    guesses.append(cm.astype(int))
            except Exception: pass
            guesses.append(ti.copy())
            out[bk] = merge_front(guesses[:1], guesses[1:])[:2]
    return out

def all_variants(decoded_results, challenges):
    """Commit-mode helper: ranked lists for every pre-registered variant, to compare on the same pools."""
    outs = {}
    for name in ("V0", "V1", "V2", "V3"):
        outs[name] = select_all(decoded_results, challenges, name)[0]
    ctrl = {}
    for bk, guesses in decoded_results.items():
        try: ctrl[bk] = v4_control_soft_delete(list(guesses.values()))
        except Exception: ctrl[bk] = outs["V0"][bk]
    outs["V4_control_soft_delete"] = ctrl
    return outs
