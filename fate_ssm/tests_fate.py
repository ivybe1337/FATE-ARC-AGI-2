"""Soundness test for FateEngine: planted true trajectory on random nondeterministic transition systems; no true state may ever be eliminated.
Also: fixed point is idempotent (re-running on the narrowed domains changes nothing) and CONTRADICTION is reported for impossible endings."""
import random
from .fate_engine import FateEngine

def make_system(rng, n_states, branching):
    T = {s: set(rng.sample(range(n_states), rng.randint(1, branching))) for s in range(n_states)}
    return T

def test_soundness(n_cases=600, seed=0):
    rng = random.Random(seed); bad = narrowed_more = 0
    for _ in range(n_cases):
        n = rng.randint(6, 40); T = make_system(rng, n, rng.randint(1, 4)); H = rng.randint(2, 7)
        traj = [rng.randrange(n)]
        for _ in range(H): traj.append(rng.choice(sorted(T[traj[-1]])))
        eng = FateEngine(lambda s: T[s], universe=range(n))
        res = eng.run(traj[0], H, {traj[-1]})
        for t, s in enumerate(traj):
            if s not in res.domains[t]: bad += 1; break
        fwd_sizes = []; cur = {traj[0]}
        for _ in range(H): cur = set().union(*[T[s] for s in cur]); fwd_sizes.append(len(cur))
        narrowed_more += sum(len(res.domains[t + 1]) < fwd_sizes[t] for t in range(H - 1)) > 0
    print(f"soundness: {n_cases} random systems, true state eliminated in {bad} cases | FATE strictly narrower than forward-only in {narrowed_more}")
    assert bad == 0
def test_contradiction():
    T = {0: {1}, 1: {2}, 2: {2}, 3: {3}}
    res = FateEngine(lambda s: T[s], universe=range(4)).run(0, 3, {3})
    assert res.status() == "CONTRADICTION", res.status(); print("impossible ending -> CONTRADICTION ok")
def test_idempotent():
    rng = random.Random(1)
    for _ in range(100):
        n = rng.randint(6, 25); T = make_system(rng, n, 3); H = 4; traj = [rng.randrange(n)]
        for _ in range(H): traj.append(rng.choice(sorted(T[traj[-1]])))
        eng = FateEngine(lambda s: T[s], universe=range(n)); r1 = eng.run(traj[0], H, {traj[-1]})
        # a second pass restricted to the first result's terminal set must give identical domains
        r2 = eng.run(traj[0], H, set(r1.domains[H])); assert [set(d) for d in r1.domains] == [set(d) for d in r2.domains]
    print("fixed point idempotent on 100 random systems")
if __name__ == "__main__":
    test_soundness(); test_contradiction(); test_idempotent(); print("ALL OK")
