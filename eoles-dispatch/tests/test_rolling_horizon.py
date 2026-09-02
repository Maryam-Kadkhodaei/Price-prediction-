"""Tests for the rolling-horizon parameters of build_model (models/default.py):
initial_on, initial_soc, committed_hours, window_hours, lake_ceiling,
thermal_ceiling.

These exist to catch a regression of a real cold-start infeasibility bug
found and fixed in this repo: the model used to force every thermal unit's
on/off state to full capacity at the first hour of a window regardless of
any ceiling on that unit, which is infeasible whenever a unit's ceiling for
its committed hours is tighter than its full-capacity default (e.g. a unit
with little to no allotted budget for that window). The fix: initial_on_rule
is skipped entirely for any (area, tech) pair with no explicit prior-window
state, leaving on[h0] free -- bounded only by the unit's normal capacity
limits, same as any other hour -- instead of pinning it to an arbitrary
default.
"""

import pyomo.environ  # noqa: F401 -- registers solver plugins
from pyomo.opt import SolverFactory, TerminationCondition

from eoles_dispatch.models import build_default_model

AREAS = ["FR", "DE"]
THR = ["nuclear", "gas_ccgt1G"]
HOURS = [0, 1, 2]


class TestInitialOnConstraint:
    def test_skipped_on_cold_start(self, input_dir):
        """With no prior-window state, on[h0] should be left unconstrained."""
        model = build_default_model(input_dir)  # initial_on defaults to None
        for a in AREAS:
            for thr in THR:
                assert (a, thr) not in model.initial_on_constraint

    def test_enforced_when_prior_state_given(self, input_dir):
        """With an explicit prior-window value, on[h0] should be pinned to it."""
        pinned = {("FR", "nuclear"): 3.0}
        model = build_default_model(input_dir, initial_on=pinned)
        assert ("FR", "nuclear") in model.initial_on_constraint
        # Any (area, tech) not covered by the handed-in state is still free.
        assert ("FR", "gas_ccgt1G") not in model.initial_on_constraint
        assert ("DE", "nuclear") not in model.initial_on_constraint


class TestColdStartFeasibility:
    def test_cold_start_with_zero_thermal_ceiling_is_feasible(self, input_dir):
        """Regression test for the cold-start infeasibility fixed in this repo.

        Before the fix, a cold start (initial_on=None) combined with a
        thermal ceiling of 0 for a unit with nonzero capacity was infeasible:
        initial_on_rule forced on[h0] to capa*maxaf (> 0) while
        yearly_maxON_constraint simultaneously required the sum over
        committed hours (including h0) to be <= 0. Skipping the constraint
        on cold start leaves on[h0] free, so the model can satisfy both.
        """
        model = build_default_model(
            input_dir,
            initial_on=None,
            committed_hours=HOURS,
            thermal_ceiling={("FR", "nuclear"): 0.0},
        )
        opt = SolverFactory("appsi_highs")
        opt.highs_options["solver"] = "ipm"
        opt.highs_options["run_crossover"] = "on"
        results = opt.solve(model)
        assert results.solver.termination_condition == TerminationCondition.optimal
