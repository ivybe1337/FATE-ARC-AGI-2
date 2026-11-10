# Pre-registration: SSM/FATE selection layer over the open NVARC candidate pools
Written 2026-10-26, before any candidate-pool or FATE-precision result was read.

## Setup (fixed now, not tuned)
Pools: raw candidates recorded from the public NVARC-based pipeline (Qwen3-4B + test-time training + DFS decoding) on ARC-AGI-2
public-evaluation tasks, scored locally against the public solutions. Each candidate grid carries votes (samples that produced it),
beam_score and 8 augmentation NLLs. Baseline score is the pipeline's own `score_kgmon` = votes - mean(mean(score_aug)).
Top-2 per test output (ARC scoring). Unit = test output. Paired comparison on identical pools.

## Variants
V0  baseline: score_kgmon (also score_full_probmul_3 reported).
V1  SSM mask (Admissibility paper, Prop. "Masking never hurts on lawful data"): candidates failing the demo-mined ledger
    (shape rule, palette, count/profile conservation; arcledger/ledger.py, byte-identical to the Commutator paper) are demoted behind
    all admissible ones. Nothing is deleted from the 2 slots unless >= 2 admissible candidates exist.
V2  FATE-forced: exact-bijection programs (fate_arc, depth 2+2, + colour map) that reproduce EVERY demo pair contribute their output as
    attempt_1 when one exists; attempt_2 from V0 order.
V3  V2 + V1.
V4  CONTROL (violates the framework's own rule "a learned scorer may order but never delete"): delete candidates whose mean
    augmentation NLL exceeds the best by more than ln(10); rank survivors by V0.

## Predictions
P1  Deterministic: V1 loses to V0 only on tasks whose true output is ledger-excluded (false exclusion). On tasks with truth admitted
    V1 >= V0 by construction.
P2  Net V1 - V0 is small: at most +2 test outputs per 100, not statistically significant at 120 tasks.
P3  FATE certified-program precision (task has a certified program -> correct test output): between 40% and 90%.
P4  V2/V3 add at most 1 correct test output per 100 over V1.
P5  V4 (soft deletion) is not better than V1.
P6  The ledger removes at least one candidate from the pool of >= 30% of test outputs.
P7  Candidate-pool recall (truth present in the pool at all) is the binding ceiling: selection variants move accuracy by far less than
    (pool recall - V0 accuracy).

## Reporting rule
Every variant, every number, losses included. Absolute accuracy is NOT claimed: the open checkpoint may have trained on public-evaluation
tasks, so only paired differences between selection rules on fixed pools are reported.
