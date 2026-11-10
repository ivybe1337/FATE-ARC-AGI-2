"""Property tests (no GPU, no data). Run: uv run --with numpy python -m fate_ssm.tests_select"""
import numpy as np
from . import select_pool as sp
from .ledger import Ledger

rng = np.random.default_rng(0)


def rand_grid(h, w, cols): return rng.choice(cols, size=(h, w)).astype(int)


def make_task():
    cols = list(rng.choice(np.arange(1, 10), 4, replace=False))
    demos = []
    for _ in range(3):
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6)); g = rand_grid(h, w, cols)
        demos.append((g, g[:, ::-1].copy()))                 # same shape, same palette, count-conserving
    ti = rand_grid(4, 5, cols); truth = ti[:, ::-1].copy()
    return demos, ti, truth, cols


def make_pool(ti, truth, cols, n_cand=8):
    cands = [truth.copy()]
    for _ in range(n_cand - 1):
        c = ti.copy() if rng.random() < .5 else rand_grid(*ti.shape, cols + [0, 9])
        if rng.random() < .3: c = rand_grid(ti.shape[0] + 1, ti.shape[1], cols)
        cands.append(c)
    pool = []
    for c in cands:
        for _ in range(int(rng.integers(1, 4))):
            pool.append(dict(solution=c, beam_score=float(rng.random()), score_aug=list(rng.random(8) * 3)))
    return pool


def test_masking_never_lowers_truth_rank_when_truth_admitted():
    checked = 0
    for _ in range(400):
        demos, ti, truth, cols = make_task(); L = Ledger(demos); pool = make_pool(ti, truth, cols)
        if not L.admits(ti, truth): continue
        r0 = sp.rank_of(sp.v0(pool), truth); r1 = sp.rank_of(sp.v1_mask(pool, L, ti)[0], truth)
        assert r1 <= r0, (r0, r1); checked += 1
    assert checked > 300, checked
    print("masking property holds on", checked, "random pools")


def test_mask_identity_when_everything_admitted():
    for _ in range(100):
        demos, ti, truth, cols = make_task(); L = Ledger(demos)
        pool = [dict(solution=truth, beam_score=.1, score_aug=[.1] * 8), dict(solution=truth[:, ::-1], beam_score=.2, score_aug=[.2] * 8)]
        a = sp.v0(pool); b, n_exc = sp.v1_mask(pool, L, ti)
        if n_exc == 0: assert all((x == y).all() for x, y in zip(a, b))
    print("mask is the identity when nothing is excluded")


def test_fate_forced_goes_first_and_dedupes():
    demos, ti, truth, cols = make_task(); pool = make_pool(ti, truth, cols)
    out = sp.v2_fate(pool, [truth, truth]); assert (out[0] == truth).all() and len(out) == len({g.tobytes() + bytes(g.shape) for g in out})
    print("FATE-forced placement and dedup ok")


def test_soft_delete_control_runs():
    demos, ti, truth, cols = make_task(); assert len(sp.v4_control_soft_delete(make_pool(ti, truth, cols))) >= 1
    print("control runs")


if __name__ == "__main__":
    test_masking_never_lowers_truth_rank_when_truth_admitted(); test_mask_identity_when_everything_admitted()
    test_fate_forced_goes_first_and_dedupes(); test_soft_delete_control_runs(); print("ALL OK")
