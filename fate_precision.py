"""How precise are FATE's certified programs on real ARC-AGI-2? (CPU only, no LLM.)
certified = at least one exact-bijection program that reproduces EVERY demo pair. precision = of tasks with a certified program,
how many have a correct test output within the 2 attempts. Everything recorded per task."""
import json, sys, time
import numpy as np
from fate_ssm import fate_arc as fa
import os; D = os.path.join(os.environ.get("ARC_DATA", "data"), "")
out = {}
for split in ("evaluation", "training"):
    ch = json.load(open(D + f"arc-agi_{split}_challenges.json")); so = json.load(open(D + f"arc-agi_{split}_solutions.json"))
    rows = []; t0 = time.time()
    for tid in sorted(ch):
        t = ch[tid]
        outs, nprog, ncm = fa.solve(t["train"], [p["input"] for p in t["test"]], 2, 2, True)
        hit = [any(np.array_equal(o, np.array(s)) for o in outs[k]) for k, s in enumerate(so[tid])]
        rows.append(dict(task=tid, n_prog=nprog, n_cmap=ncm, certified=bool(nprog or ncm), all_hit=all(hit), hits=sum(hit), n_tests=len(hit)))
    cert = [r for r in rows if r["certified"]]
    out[split] = dict(n_tasks=len(rows), certified_tasks=len(cert), certified_and_correct=sum(r["all_hit"] for r in cert),
                      cases_hit=sum(r["hits"] for r in rows), cases=sum(r["n_tests"] for r in rows), seconds=round(time.time()-t0))
    json.dump(rows, open(f"receipts/fate_precision_{split}.json", "w"))
    print(split, out[split], flush=True)
json.dump(out, open("receipts/fate_precision_summary.json", "w"), indent=1)
