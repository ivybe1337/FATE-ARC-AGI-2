"""
FATE Engine — Bidirectional Elimination-Based Inference Architecture
=====================================================================

Core idea (see FATE_Engine_Technical_Specification.md for the full write-up):

Most sequence-prediction models are one-directional: given a start state, a forward
generative/transition model proposes what could come next, and that's the whole prediction.
This leaves ambiguity whenever the transition model is not deterministic — many different
futures remain "possible" even though, if you already know or hypothesize something about a
later state (a goal, an observation, a terminal constraint), most of those futures are actually
impossible.

FATE Engine runs inference in BOTH directions and intersects the results at every time step,
then repeats until nothing more can be eliminated (a fixed point):

  1. FORWARD PASS  — from the known/assumed initial state, compute the set of states reachable
     at each future time step under the transition model (ordinary forward generation).
  2. BACKWARD PASS — from a known/hypothesized terminal (future) constraint, compute the set of
     states at each earlier time step that COULD possibly lead to that terminal constraint
     (the "preimage" under the transition model — the same mathematics as the adjoint/costate
     backward pass in control theory, but over discrete states instead of continuous ones).
  3. INTERSECT     — a state survives at time t only if it is BOTH forward-reachable from the
     start AND backward-necessary for the (hypothesized) end. Anything else is eliminated and
     the elimination is logged with its exact reason (a "certificate" / "nogood receipt").
  4. REPEAT        — narrowing one time step can shrink what's reachable/necessary at neighboring
     steps, so passes 1-3 repeat until no domain changes (a standard fixed-point / arc-consistency
     loop, same structural idea as AC-3 constraint propagation and the classical HMM
     forward-backward algorithm, generalized to arbitrary discrete transition relations and
     applied here to trajectory / sequence inference).

This is a fully causal, ordinary computation — nothing here requires or claims literal
foreknowledge. The "future" input to the backward pass is just data: a known ending, a stated
goal, an observed later frame, or a hypothesis under test. What makes this a genuinely different
*architecture* from standard forward-only generation is combining a flexible forward proposal
mechanism with certified backward necessity pruning, and reporting *why* each survivor survived
and each elimination happened — which a purely forward generative model cannot do at all.

See FATE_Engine_Technical_Specification.md, Section 4, for the honest scope of what "novel" means
here (a new combination of established techniques, not a new fundamental principle) and its
relationship to the philosophical FATE_Teleological_Framework.md, which this engine's *mechanism*
supports but does not require you to believe.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, Any, Optional
import itertools


State = Any  # any hashable object representing a discrete state


@dataclass
class Elimination:
    """A single certified removal, with its exact cause — the receipt/certificate chain."""
    time_step: int
    state: State
    reason: str


@dataclass
class FateResult:
    """Final narrowed domains at every time step, plus the full elimination log."""
    domains: list[set[State]]
    eliminations: list[Elimination]
    horizon: int

    def status(self) -> str:
        if any(len(d) == 0 for d in self.domains):
            return "CONTRADICTION"          # the hypothesized future is impossible given dynamics
        if all(len(d) == 1 for d in self.domains):
            return "FULLY_DETERMINED"       # unique trajectory certified
        return "NARROWED"                   # reduced uncertainty, not fully unique

    def trajectory(self) -> Optional[list[State]]:
        if self.status() != "FULLY_DETERMINED":
            return None
        return [next(iter(d)) for d in self.domains]

    def uncertainty_profile(self) -> list[int]:
        """Domain size at each time step — the direct, measurable 'how much did we narrow it' metric."""
        return [len(d) for d in self.domains]

    def explain(self, time_step: int, state: State) -> list[str]:
        """Every certified reason a specific (time, state) pair was eliminated, in order."""
        return [e.reason for e in self.eliminations if e.time_step == time_step and e.state == state]


class FateEngine:
    """
    Generic bidirectional elimination engine over a chain of T+1 time steps (0..T).

    transition_fn(state) -> set of possible next states.  This is the "forward proposal"
        mechanism. It can be a hand-written rule set (as in the demo) or, in a real deployment,
        the candidate set produced by a learned generative model (e.g. top-k decode).

    inverse_transition_fn(state) -> set of possible previous states, i.e. the preimage:
        "which states could have transitioned into this one." If not supplied, it is derived
        automatically by brute-force inversion over a supplied universe (fine for small/discrete
        domains; a real deployment would supply an exact or learned inverse instead).
    """

    def __init__(
        self,
        transition_fn: Callable[[State], set[State]],
        universe: Optional[Iterable[State]] = None,
        inverse_transition_fn: Optional[Callable[[State], set[State]]] = None,
        nogoods: Optional[list[Callable[[int, State], Optional[str]]]] = None,
    ):
        self.transition_fn = transition_fn
        self.universe = list(universe) if universe is not None else None
        self._inverse_fn = inverse_transition_fn
        # nogoods: functions (time_step, state) -> reason string if INADMISSIBLE, else None.
        # These are the certified, hard, per-state exclusion rules (e.g. domain bounds, forbidden
        # patterns) — the discrete analogue of CSM's "structural exclusion."
        self.nogoods = nogoods or []

    def _inverse(self, state: State) -> set[State]:
        if self._inverse_fn is not None:
            return self._inverse_fn(state)
        if self.universe is None:
            raise ValueError(
                "No inverse_transition_fn supplied and no universe given to brute-force invert from."
            )
        return {s for s in self.universe if state in self.transition_fn(s)}

    def _apply_nogoods(self, t: int, domain: set[State], log: list[Elimination]) -> set[State]:
        survivors = set()
        for s in domain:
            reason = None
            for rule in self.nogoods:
                reason = rule(t, s)
                if reason:
                    break
            if reason:
                log.append(Elimination(time_step=t, state=s, reason=reason))
            else:
                survivors.add(s)
        return survivors

    def run(
        self,
        initial_state: State,
        horizon: int,
        terminal_constraint: set[State],
        max_iterations: int = 50,
    ) -> FateResult:
        """
        initial_state:        the known state at t=0.
        horizon:               number of steps forward (domains computed for t=0..horizon).
        terminal_constraint:   the set of states admissible at t=horizon — the known/hypothesized
                                "future" input to the backward pass. Pass the single true known
                                ending as a singleton set, or a broader hypothesis set if the
                                ending is only partially known.
        """
        log: list[Elimination] = []
        domains: list[set[State]] = [set() for _ in range(horizon + 1)]
        domains[0] = {initial_state}
        domains[horizon] = set(terminal_constraint)

        # Seed initial forward/backward passes so every interior step has *some* domain
        # before the fixed-point loop starts narrowing them further.
        fwd = [domains[0]]
        for t in range(1, horizon + 1):
            nxt = set()
            for s in fwd[-1]:
                nxt |= self.transition_fn(s)
            fwd.append(nxt)

        bwd = [None] * (horizon + 1)
        bwd[horizon] = domains[horizon]
        for t in range(horizon - 1, -1, -1):
            prev = set()
            for s in bwd[t + 1]:
                prev |= self._inverse(s)
            bwd[t] = prev

        for t in range(horizon + 1):
            candidate = fwd[t] & bwd[t] if t not in (0, horizon) else domains[t]
            eliminated_by_intersection = (fwd[t] | bwd[t]) - candidate
            for s in eliminated_by_intersection:
                in_fwd, in_bwd = s in fwd[t], s in bwd[t]
                reason = (
                    "not forward-reachable from the initial state"
                    if not in_fwd
                    else "not backward-necessary for the terminal constraint"
                )
                log.append(Elimination(time_step=t, state=s, reason=reason))
            domains[t] = candidate

        domains[0] = {initial_state}

        # Fixed-point loop: apply nogoods, then re-propagate forward/backward from the
        # NARROWED neighbors, repeating until nothing changes (or contradiction / max_iterations).
        for iteration in range(max_iterations):
            changed = False

            for t in range(horizon + 1):
                pruned = self._apply_nogoods(t, domains[t], log)
                if pruned != domains[t]:
                    domains[t] = pruned
                    changed = True

            if any(len(d) == 0 for d in domains):
                break  # contradiction — no point continuing to propagate an empty domain

            for t in range(1, horizon + 1):
                allowed_next = set()
                for s in domains[t - 1]:
                    allowed_next |= self.transition_fn(s)
                new_domain = domains[t] & allowed_next
                if new_domain != domains[t]:
                    for s in domains[t] - new_domain:
                        log.append(Elimination(
                            time_step=t, state=s,
                            reason=f"no admissible predecessor survives at t={t-1}",
                        ))
                    domains[t] = new_domain
                    changed = True

            for t in range(horizon - 1, -1, -1):
                allowed_prev = set()
                for s in domains[t + 1]:
                    allowed_prev |= self._inverse(s)
                new_domain = domains[t] & allowed_prev
                if new_domain != domains[t]:
                    for s in domains[t] - new_domain:
                        log.append(Elimination(
                            time_step=t, state=s,
                            reason=f"no admissible successor survives at t={t+1}",
                        ))
                    domains[t] = new_domain
                    changed = True

            if not changed:
                break

        return FateResult(domains=domains, eliminations=log, horizon=horizon)
