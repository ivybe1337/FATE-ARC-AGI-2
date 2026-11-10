"""Forced-cell constraints mined from demos (SSM 'forced value' calculus applied to local input->output rules).
For same-shape tasks: key = (radius-r neighbourhood of the INPUT around a cell); the demos give the set of output colours seen for that key.
If a key was seen and always mapped to ONE colour, the test cell with that key is FORCED to that colour (assumption: the task is a local rule).
violations(candidate) = number of forced cells the candidate contradicts. Coverage/precision are measured on the TRUE outputs."""
from __future__ import annotations
import numpy as np
PAD = 10

def _keys(g, r):
    H, W = g.shape; p = np.pad(g, r, constant_values=PAD); K = []
    for y in range(H):
        for x in range(W):
            K.append(p[y:y + 2 * r + 1, x:x + 2 * r + 1].tobytes())
    return K

def mine(demos, r):
    """demos: [(inp, out)] all same shape pairs. returns dict key -> set(colours)"""
    T = {}
    for i, o in demos:
        for k, c in zip(_keys(i, r), o.ravel()):
            T.setdefault(k, set()).add(int(c))
    return T

def forced_cells(T, test_in, r):
    """returns array same shape as test_in: forced colour or -1"""
    out = np.full(test_in.shape, -1, int); ks = _keys(test_in, r)
    for idx, k in enumerate(ks):
        s = T.get(k)
        if s is not None and len(s) == 1: out.flat[idx] = next(iter(s))
    return out

def violations(forced, cand):
    if cand.shape != forced.shape: return None
    m = forced >= 0
    return int((cand[m] != forced[m]).sum())
