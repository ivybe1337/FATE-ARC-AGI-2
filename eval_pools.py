"""Offline evaluation of the pre-registered selection variants (PREREG.md) over recorded candidate pools.
usage: python eval_pools.py <pools_dir> [tag]
Writes receipts/pools_<tag>_rows.jsonl (one row per test output, every variant) and receipts/pools_<tag>_summary.json."""
import bz2, json, os, pickle, sys, math
import numpy as np
from fate_ssm import select_pool as sp, fate_arc as fa
from fate_ssm.ledger import Ledger

import os; D = os.path.join(os.environ.get("ARC_DATA", "data"), "")
pools_dir = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "run"
ch = json.load(open(D + "arc-agi_evaluation_challenges.json")); so = json.load(open(D + "arc-agi_evaluation_solutions.json"))

pools = {}
for fn in sorted(os.listdir(pools_dir)):
    with bz2.BZ2File(os.path.join(pools_dir, fn)) as f: samples = pickle.load(f)
    pools.setdefault(fn.split(".")[0], []).extend(samples)

fate_cache = {}
def fate_outs(tid):
    if tid not in fate_cache:
        t = ch[tid]; outs, nprog, ncm = fa.solve(t["train"], [p["input"] for p in t["test"]], 2, 2, True)
        fate_cache[tid] = outs if (nprog or ncm) else None
    return fate_cache[tid]

VARS = ["V0_kgmon", "V0_probmul3", "V1_mask", "V2_fate", "V3_mask+fate", "V4_control_soft_delete"]
rows = []
for bk, samples in sorted(pools.items()):
    tid, idx = bk.split("_"); idx = int(idx); task = ch[tid]; truth = np.array(so[tid][idx]); ti = np.array(task["test"][idx]["input"])
    demos = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]; L = Ledger(demos)
    F = fate_outs(tid); Fi = F[idx] if F is not None else []
    r0 = sp.v0(samples); r0p = sp.v0(samples, sp.probmul3); r1, n_exc = sp.v1_mask(samples, L, ti)
    r2 = sp.v2_fate(samples, Fi); r3 = sp.v3_both(samples, L, ti, Fi); r4 = sp.v4_control_soft_delete(samples)
    ranked = dict(zip(VARS, [r0, r0p, r1, r2, r3, r4]))
    G = sp.group(samples)
    row = dict(key=bk, n_candidates=len(G), n_samples=len(samples), truth_in_pool=sp.rank_of(r0, truth) is not None,
               truth_admitted=bool(L.admits(ti, truth)), ledger_excluded=n_exc, fate_certified=bool(Fi), fate_correct=bool(Fi and sp.top2_hit(Fi, truth)),
               rank_v0=sp.rank_of(r0, truth), rank_v1=sp.rank_of(r1, truth))
    for v in VARS: row[v] = bool(sp.top2_hit(ranked[v], truth))
    rows.append(row)

with open(f"receipts/pools_{tag}_rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")

def sign_test(a, b):
    w = sum(x and not y for x, y in zip(a, b)); l = sum(y and not x for x, y in zip(a, b)); n = w + l
    if n == 0: return w, l, 1.0
    k = min(w, l); p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n); return w, l, p

n = len(rows); S = dict(n_test_outputs=n, n_tasks=len({r["key"].split("_")[0] for r in rows}), tag=tag,
    pool_recall=round(sum(r["truth_in_pool"] for r in rows) / max(n, 1), 4), truth_admitted=sum(r["truth_admitted"] for r in rows),
    outputs_with_ledger_exclusion=sum(r["ledger_excluded"] > 0 for r in rows), fate_certified=sum(r["fate_certified"] for r in rows),
    fate_certified_correct=sum(r["fate_correct"] for r in rows), accuracy={}, paired_vs_V0={})
for v in VARS: S["accuracy"][v] = dict(correct=sum(r[v] for r in rows), of=n, acc=round(sum(r[v] for r in rows) / max(n, 1), 4))
base = [r["V0_kgmon"] for r in rows]
for v in VARS[1:]:
    w, l, p = sign_test([r[v] for r in rows], base); S["paired_vs_V0"][v] = dict(wins=w, losses=l, sign_test_p=round(p, 4))
viol = [r for r in rows if r["rank_v1"] is not None and r["rank_v0"] is not None and r["rank_v1"] > r["rank_v0"]]
S["P1_violations_truth_rank_got_worse_even_though_admitted"] = sum(1 for r in viol if r["truth_admitted"])
S["truth_rank_worse_under_mask_total"] = len(viol)
json.dump(S, open(f"receipts/pools_{tag}_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
