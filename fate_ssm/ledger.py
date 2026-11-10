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
