# FATE: ARC Prize 2026, ARC-AGI-2

This is my code submission for the ARC-AGI-2 track of ARC Prize 2026.

| | |
|---|---|
| Kaggle notebook | [netphantom17/fate-ssm-final](https://www.kaggle.com/code/netphantom17/fate-ssm-final) |
| Submission | 57913192, public score **29.72** |
| Earlier submission, same layer | 57677356, public score 28.06 |
| Paper | [*Likely and Allowed*](https://ivybe1337.github.io/Survival-Mechanics/), DOI [10.5281/zenodo.24093939](https://doi.org/10.5281/zenodo.24093939) |

## The idea

A model's likelihood tells you what is probable. It does not tell you what the task allows. FATE keeps those two jobs apart.

The candidate answers come from the open NVARC pipeline: Qwen3-4B (`sorokin/qwen3_4b_grids15_sft139`), per-task LoRA test-time training, DFS decoding and augmentation re-scoring. I use it unmodified. It is not my work and the score belongs to the pipeline and my layer together.

My part is the final selection, under one rule:

> The model's score may **order** candidates. Only an admissibility rule mined from the demonstrations may **demote** them. Nothing is ever deleted.

The admissibility rule is a ledger read off the demonstration pairs:

1. the output shape rule (same as the input, or transposed),
2. the palette (which colours an output may introduce),
3. the exact colour count,
4. the sorted colour profile.

An invariant enters the ledger only if every demonstration pair obeys it. A candidate that breaks the ledger is moved behind every candidate that keeps it. If fewer than two admissible candidates exist, the two attempt slots still fill from the model's order, so the layer can never empty a slot.

The submitted variant is **V1** (ledger demotion). The variants were fixed in [`PREREG.md`](PREREG.md) before any candidate pool or precision result was read:

| Variant | Rule |
|---|---|
| V0 | the pipeline's own score, unchanged |
| V1 | V0 order, ledger-inadmissible candidates demoted |
| V2 | a FATE-certified program, if one exists, fills attempt 1 |
| V3 | V2 and V1 together |
| V4 | control: delete by likelihood threshold, which the rule forbids |

## FATE, the certified program search

`fate_ssm/fate_arc.py` and `fate_ssm/fate_engine.py` run an exact meet-in-the-middle search over grid bijections (depth 2 from each side, plus a colour map). A program is **certified** only if it reproduces every demonstration pair exactly. A certified program's output is a positive certificate; a candidate the ledger rules out is a certified exclusion. Everything else stays unknown and is ordered by the model.

Measured precision of certified programs (`receipts/fate_precision_summary.json`): on the 1,000 training tasks, 15 tasks certify and 14 of those are correct. On the 120 evaluation tasks, none certify. That is why the submitted variant is V1, not V2.

## Terms

The notebook was written in my working vocabulary. The paper's names:

| Notebook | Paper |
|---|---|
| Survival State Mechanics (SSM) | the joint state of prediction and admissibility |
| survival state κ | residual state κ |
| survival semantics; the demonstration ledger | admissibility rule V |
| model score, candidate likelihood | predictive state ε and its forecast |
| SSM mask (V1) | ordering by ε, demotion by V |
| FATE-certified program (V2) | certified admissible, status A |
| nogood, certified elimination | certified exclusion, status X |

## Layout

```
notebook/                 the submitted Kaggle notebook and its metadata
fate_ssm/                 the selection layer and FATE, as a package
  ledger.py               the demonstration ledger
  select_pool.py          variants V0 to V4 over a candidate pool
  fate_arc.py             FATE program search on ARC grids
  fate_engine.py          the meet-in-the-middle engine
  local_rule.py           local cell rules (offline study)
  entry.py                the notebook entry points
  tests_fate.py           tests for FATE
  tests_select.py         tests for the selection layer
build_runtime.py          concatenates the package into fate_ssm_runtime.py
fate_ssm_runtime.py       the single file the notebook writes to disk
make_final_kernel.py      builds the notebook from the NVARC baseline
eval_pools.py             offline comparison of the variants on recorded pools
fate_precision.py         precision of certified programs
local_rule_stats.py       coverage and precision of local rules
receipts/                 the precision and local-rule results
PREREG.md                 the pre-registration
```

## Running

```
pip install numpy scipy
python -m fate_ssm.tests_fate
python -m fate_ssm.tests_select
export ARC_DATA=/path/to/arc-prize-2026-arc-agi-2/
python fate_precision.py
```

`ARC_DATA` is the folder holding the competition's `arc-agi_*_challenges.json` and `arc-agi_*_solutions.json`. The full pipeline needs a GPU and runs on Kaggle; the notebook is the reference.

## License

My code in this repository is released under [MIT-0](LICENSE). The NVARC pipeline cells in the notebook and `sorokin/pip-install-unsloth-flash-patch` belong to their authors and stay under their terms.

Joshua Matthew Carter
