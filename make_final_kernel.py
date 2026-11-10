"""Builds the PRIVATE submission-candidate notebook: the proven open NVARC pipeline (cells 1-7 untouched) + the SSM/FATE selection layer.
usage: python make_final_kernel.py <user> <name> <VARIANT: V0|V1|V2|V3> [--eval-subset N]
In commit (non-rerun) mode the pipeline runs on a small seeded subset of public-eval tasks (so the commit run is short and prints a baseline-vs-ours comparison);
in competition-rerun mode it processes the full hidden test set exactly like the baseline and then applies the selection layer with fallbacks."""
import json, os, random, sys
user, name, variant = sys.argv[1:4]; n_sub = int(sys.argv[sys.argv.index("--eval-subset") + 1]) if "--eval-subset" in sys.argv else 4
SRC = os.environ["NVARC_SRC"]  # folder holding the baseline notebook and its kernel-metadata.json
D = os.path.join(os.environ.get("ARC_DATA", "data"), "arc-agi_evaluation_challenges.json")
_ids = sorted(json.load(open(D))); tasks = _ids if n_sub <= 0 else random.Random(0).sample(_ids, n_sub)
nb = json.load(open(f"{SRC}/arc-agi2-lb33-89-minimal-perfpatch.ipynb")); meta = json.load(open(f"{SRC}/kernel-metadata.json"))
cell = lambda src: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.splitlines(keepends=True))
setsrc = lambda i, s: nb["cells"].__setitem__(i, cell(s)) if nb["cells"][i]["cell_type"] == "code" else nb["cells"][i].__setitem__("source", s.splitlines(keepends=True))
src = lambda i: "".join(nb["cells"][i]["source"])
setsrc(0, f"# FATE/SSM selection layer over the open NVARC pipeline (private candidate)\nVariant {variant}. Cells 1-7 are the unmodified baseline pipeline; only the final selection differs.\n")
st = src(6); old = '            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:\n                continue\n'
assert old in st; setsrc(6, st.replace(old, f'            if key not in {json.dumps(tasks)}:\n                continue\n'))
runtime = open("fate_ssm_runtime.py").read()
final = f'''import os, json, shutil
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
import fate_ssm_runtime as R

VARIANT = "{variant}"
rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
CH = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_%s_challenges.json" % ("test" if rerun_mode else "evaluation")
data = ArcDataset.from_file(CH)
if not rerun_mode:
    data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")
challenges = json.load(open(CH))
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
decoder.load_decoded_results("/kaggle/inference_outputs")
base_sel = decoder.run_selection_algo()                       # the pipeline's own selection, always computed
final = base_sel
try:
    sel, stats = R.select_all(decoder.decoded_results, challenges, VARIANT)
    fb = R.fallback_for_missing(challenges, set(decoder.decoded_results))
    final = dict(fb); final.update(sel)
    print("selection layer:", VARIANT, stats, "| CPU-fallback outputs for unreached tasks:", len(fb))
except Exception:
    import traceback; traceback.print_exc(); print("selection layer failed -> baseline selection used"); final = base_sel
submission = data.get_submission(final)
assert set(submission) == set(challenges) and all(len(v) == len(challenges[k]["test"]) for k, v in submission.items())
with open("submission.json", "w") as f: json.dump(submission, f)
if not rerun_mode:
    covered = {{k.split("_")[0] for k in decoder.decoded_results}}
    only = lambda sub: {{k: v for k, v in sub.items() if k in covered}}
    def score(sub):
        s = 0
        for k, v in data.replies.items():
            if k in covered:
                for i, r in enumerate(v):
                    if any(np.array_equal(r, sub[k][i][a]) for a in ("attempt_1", "attempt_2")): s += 1 / len(v)
        return s
    print("*** processed tasks:", len(covered), "| baseline selection:", round(score(data.get_submission(base_sel)), 3), "| ours:", round(score(submission), 3))
    shutil.copytree("/kaggle/inference_outputs", "/kaggle/working/pools", dirs_exist_ok=True)
    try:
        import math
        AV = R.all_variants(decoder.decoded_results, challenges)
        def hit(ranked, truth): return any(g.shape == np.array(truth).shape and (g == np.array(truth)).all() for g in ranked[:2])
        keys = sorted(decoder.decoded_results); res = {{}}
        for vn, sel in AV.items():
            res[vn] = [hit(sel[k], data.replies[k.split("_")[0]][int(k.split("_")[1])]) for k in keys]
        base = res["V0"]; S = dict(n_outputs=len(keys), n_tasks=len(covered))
        S["pool_recall"] = sum(any(g.shape == np.array(data.replies[k.split("_")[0]][int(k.split("_")[1])]).shape and (g == np.array(data.replies[k.split("_")[0]][int(k.split("_")[1])])).all() for g in AV["V0"][k]) for k in keys) / max(len(keys), 1)
        S["variants"] = {{}}
        for vn, h in res.items():
            w = sum(a and not b for a, b in zip(h, base)); l = sum(b and not a for a, b in zip(h, base)); n = w + l
            p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)
            S["variants"][vn] = dict(correct=sum(h), wins_vs_V0=w, losses_vs_V0=l, sign_test_p=round(p, 4))
        print("*** VARIANT COMPARISON", json.dumps(S))
        json.dump(S, open("/kaggle/working/variants_summary.json", "w"), indent=1)
    except Exception:
        import traceback; traceback.print_exc()
'''
nb["cells"].insert(3, cell("%%writefile fate_ssm_runtime.py\n" + runtime))
setsrc(len(nb["cells"]) - 1, final)
out = os.path.join("kaggle", name); os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/{name}.ipynb", "w"), indent=1)
meta.pop("id_no", None); meta.update(id=f"{user}/{name}", title=name.replace("-", " "), code_file=f"{name}.ipynb", is_private=True, enable_internet=False, enable_gpu=True)
json.dump(meta, open(f"{out}/kernel-metadata.json", "w"), indent=1)
print("built", out, "| variant", variant, "| private", meta["is_private"], "| dev subset", tasks)
