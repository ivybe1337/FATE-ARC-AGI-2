"""Selection layers over recorded candidate pools (see PREREG.md for the pre-registered definitions).

A pool is a list of samples {solution: ndarray, beam_score: float, score_aug: [8 NLLs]} for ONE test output.
Every function returns an ordered list of candidate grids (ndarray); the caller takes the first two.

Framework rule enforced here (FATE spec sec.2 + Admissibility paper): a learned/soft score may ORDER candidates, only a
declared admissibility invariant may DEMOTE one, and nothing is dropped from a slot unless >= 2 admissible candidates exist.
"""
from __future__ import annotations
import math
from typing import Callable, Optional
import numpy as np
from .ledger import Ledger

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
