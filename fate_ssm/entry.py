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
