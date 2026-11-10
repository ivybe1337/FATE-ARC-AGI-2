import json, numpy as np
from fate_ssm import local_rule as lr
import os; D = os.path.join(os.environ.get("ARC_DATA", "data"), "")
res = {}
for split in ("training", "evaluation"):
    ch = json.load(open(D + f"arc-agi_{split}_challenges.json")); so = json.load(open(D + f"arc-agi_{split}_solutions.json"))
    for r in (0, 1):
        tot = forced = right = tasks = tasks_any = tasks_viol = 0; viol_hist = {}
        rows = []
        for tid in sorted(ch):
            t = ch[tid]; demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
            if not all(i.shape == o.shape for i, o in demos): continue
            T = lr.mine(demos, r); tasks += 1; any_f = False; tv = 0
            for k, s in enumerate(so[tid]):
                ti = np.array(t["test"][k]["input"]); truth = np.array(s)
                if truth.shape != ti.shape: continue
                F = lr.forced_cells(T, ti, r); v = lr.violations(F, truth)
                tot += truth.size; nf = int((F >= 0).sum()); forced += nf; right += nf - v; any_f |= nf > 0; tv += v
                rows.append(dict(task=tid, test=k, cells=int(truth.size), forced=nf, violations=int(v)))
            tasks_any += any_f; tasks_viol += tv > 0
            viol_hist[tv] = viol_hist.get(tv, 0) + 1
        res[f"{split}|r={r}"] = dict(same_shape_tasks=tasks, cells=tot, forced_cells=forced, forced_cov=round(forced / tot, 4),
            forced_precision=round(right / max(forced, 1), 5), tasks_with_any_forced=tasks_any, tasks_truth_violates=tasks_viol,
            viol_hist={k: v for k, v in sorted(viol_hist.items())[:8]})
        json.dump(rows, open(f"receipts/local_rule_{split}_r{r}.json", "w"))
        print(f"{split:10s} r={r}", res[f"{split}|r={r}"], flush=True)
json.dump(res, open("receipts/local_rule_summary.json", "w"), indent=1)
