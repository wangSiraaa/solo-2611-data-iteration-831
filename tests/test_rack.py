from fractions import Fraction as F

import pytest

from prepquant.rack import (
    BottlePlacement,
    Condition,
    Rack,
    RackError,
    Weight,
    placements_from_manifest,
    solve_across_racks,
    solve_rack,
)


def rep(name, k_lo, k_hi, a, b):
    return Condition.build(name, k_lo, k_hi, a, b)


# --------------------------------------------------- the spec's worked example
def test_spec_example_requires_the_only_weight():
    # empty frame/shell, bottle at origin, load mass 2, vertical moment 4 (cg 2)
    rack = Rack.build("R", [{"name": "h", "x": "0", "height": "0", "allowed_weights": ["2"]}])
    placements = [BottlePlacement.build("B", 0, 0, 2, 4)]
    weights = [Weight("w", F(2))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -1, 1)])
    assert sol.feasible is True
    assert sol.added_mass == 2
    assert sol.placements == [
        {"weight": "w", "mass": "2", "hole": "h", "x": "0", "height": "0"}
    ]
    # R(0)=0, R(1)=1 -> worst (extreme) landing is 1
    assert sol.landing["c"]["landing_at_k_lo"] == "0"
    assert sol.landing["c"]["landing_at_k_hi"] == "1"


def test_spec_example_widened_boundary_needs_no_weight():
    rack = Rack.build("R", [{"name": "h", "x": "0", "height": "0", "allowed_weights": ["2"]}])
    placements = [BottlePlacement.build("B", 0, 0, 2, 4)]
    weights = [Weight("w", F(2))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -2, 2)])
    assert sol.feasible is True
    assert sol.added_mass == 0
    assert sol.placements == []
    assert sol.landing["c"]["landing_at_k_hi"] == "2"


def test_spec_example_corrected_bottle_needs_no_weight():
    # correction: area 1 -> liquid moment 2, cg 1; same narrow boundary [-1,1]
    rack = Rack.build("R", [{"name": "h", "x": "0", "height": "0", "allowed_weights": ["2"]}])
    placements = [BottlePlacement.build("B", 0, 0, 2, 2)]
    weights = [Weight("w", F(2))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -1, 1)])
    assert sol.feasible is True
    assert sol.added_mass == 0
    assert sol.landing["c"]["landing_at_k_hi"] == "1"


# --------------------------------------- extreme parameters, not just endpoints
def test_reports_both_extremes_of_each_condition():
    rack = Rack.build("R", [{"name": "h", "x": "1", "height": "0", "allowed_weights": ["1"]}])
    placements = []
    weights = [Weight("w", F(1))]
    # two conditions with different tilt intervals sharing one layout
    conds = [rep("morning", 0, 2, -1, 5), rep("evening", "-1", 0, -3, 3)]
    sol = solve_rack(rack, placements, weights, conds)
    assert sol.feasible is True
    # mass 1 at x=1, V=0 -> R(k)=1 for every k, always inside both boundaries
    assert sol.landing["morning"]["landing_at_k_lo"] == "1"
    assert sol.landing["morning"]["landing_at_k_hi"] == "1"
    assert sol.landing["evening"]["landing_at_k_lo"] == "1"
    assert sol.landing["evening"]["landing_at_k_hi"] == "1"


# -------------------------------------- two holes, allocation changes the answer
def test_two_holes_allocation_matters():
    # One weight of mass 2; two holes at x=-2 and x=+2.  The load has no
    # horizontal moment, vertical moment 4, mass 2.  Condition [-1,1], k in [0,1]
    # -> base landing at k=1 is 2 (too far right).  Putting the weight in the
    # LEFT hole pulls it back into range; the RIGHT hole makes it worse.  Only a
    # particular allocation works, so the solver must choose correctly.
    rack = Rack.build(
        "R",
        [
            {"name": "left", "x": "-2", "height": "0"},
            {"name": "right", "x": "2", "height": "0"},
        ],
    )
    placements = [BottlePlacement.build("B", 0, 0, 2, 4)]
    weights = [Weight("w", F(2))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -1, 1)])
    # weight must go left: H = 2*(-2) = -4, M = 4, V = 4 -> R(1) = 0, R(0) = -1
    assert sol.feasible is True
    assert sol.added_mass == 2
    assert sol.placements[0]["hole"] == "left"
    assert sol.landing["c"]["landing_at_k_lo"] == "-1"
    assert sol.landing["c"]["landing_at_k_hi"] == "0"


def test_two_holes_only_one_assignment_feasible_reports_it():
    # Symmetric variant where the RIGHT hole is the only feasible choice.
    rack = Rack.build(
        "R",
        [
            {"name": "left", "x": "-2", "height": "0"},
            {"name": "right", "x": "2", "height": "0"},
        ],
    )
    placements = [BottlePlacement.build("B", 0, 0, 2, -4)]  # negative vertical moment
    weights = [Weight("w", F(2))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -1, 1)])
    # base R(0)=0, R(1)=(-4)/2=-2 (too left).  Right hole: H=4 -> R(0)=1, R(1)=0.
    assert sol.feasible is True
    assert sol.placements[0]["hole"] == "right"
    assert sol.landing["c"]["landing_at_k_lo"] == "1"
    assert sol.landing["c"]["landing_at_k_hi"] == "0"


# -------------------------------------------------------- hole allow-lists / mass
def test_weight_only_allowed_in_specific_hole():
    rack = Rack.build(
        "R",
        [
            {"name": "a", "x": "0", "height": "0", "allowed_weights": ["5"]},
            {"name": "b", "x": "-3", "height": "0", "allowed_weights": ["2"]},
        ],
    )
    placements = [BottlePlacement.build("B", 0, 0, 1, 3)]
    weights = [Weight("w5", F(5)), Weight("w2", F(2))]
    # base: M=1, V=3 -> R(0)=0, R(1)=3; boundary [-2,0] makes R(1) too far right.
    # w5 is only allowed at a (x=0) and cannot move H; w2 only at b (x=-3) does.
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -2, 0)])
    assert sol.feasible is True
    used = {p["weight"]: p["hole"] for p in sol.placements}
    assert used == {"w2": "b"}
    assert sol.added_mass == 2
    # w2 at b: H=-6, M=3, V=3 -> R(0)=-2, R(1)=-1
    assert sol.landing["c"]["landing_at_k_lo"] == "-2"
    assert sol.landing["c"]["landing_at_k_hi"] == "-1"


def test_minimum_added_mass_prefers_light_weight():
    # Two weights (1 and 3) both allowed everywhere; need a small nudge at x=-2.
    rack = Rack.build(
        "R",
        [{"name": "h1", "x": "-2", "height": "0"}, {"name": "h2", "x": "0", "height": "0"}],
    )
    placements = [BottlePlacement.build("B", 0, 0, 4, 8)]
    weights = [Weight("light", F(1)), Weight("heavy", F(3))]
    # base: M=4,V=8 -> R(1)=2 outside [-1,1]; light at h1: M=5,H=-2,V=8 -> R(1)=6/5 >1;
    # need both? heavy at h1: M=7,H=-6 -> R(1)=2/7, R(0)=-6/7 inside. heavy=3.
    # light+heavy at h1 only one per hole; light at h1 and heavy at h2:
    #   M=8,H=-2,V=8 -> R(1)=3/4, R(0)=-1/4 inside. added 4 > 3. So heavy(3) wins.
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -1, 1)])
    assert sol.feasible is True
    assert sol.added_mass == 3
    assert sol.placements[0]["weight"] == "heavy"
    assert sol.placements[0]["hole"] == "h1"


# --------------------------------------------------------------- infeasibility
def test_infeasible_reported_honestly():
    rack = Rack.build("R", [{"name": "h", "x": "0", "height": "0", "allowed_weights": ["2"]}])
    placements = [BottlePlacement.build("B", 0, 0, 2, 4)]
    weights = [Weight("w", F(2))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, 2, 3)])
    assert sol.feasible is False
    assert sol.added_mass is None
    assert sol.reason


def test_empty_hole_never_used_twice():
    rack = Rack.build("R", [{"name": "only", "x": "-1", "height": "0"}])
    placements = [BottlePlacement.build("B", 2, 0, 1, 0)]   # base M=1,H=2 -> R=2
    weights = [Weight("a", F(1)), Weight("b", F(1))]
    # boundary [0,1]: base R=2 outside; one weight at x=-1 gives M=2,H=1 -> R=1/2.
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 0, 0, 1)])
    assert sol.feasible is True
    assert sol.added_mass == 1
    assert len(sol.placements) == 1            # the single hole holds one weight


# ------------------------------------------------------------- frame mass/moment
def test_frame_mass_and_moment_counted():
    rack = Rack.build(
        "R",
        [{"name": "h", "x": "1", "height": "0"}],
        frame_mass=3,
        frame_moment_x=3,
        frame_moment_z=0,
    )
    # base M=3,H=3,V=0 -> R(k)=1 always, no bottle, no weight needed for [-1,1]
    sol = solve_rack(rack, [], [], [rep("c", 0, 1, -1, 1)])
    assert sol.feasible is True
    assert sol.added_mass == 0
    assert sol.base == {"mass": "3", "H": "3", "V": "0"}


def test_bottle_moment_folding_hand_check():
    # bottle mass 2, own vertical moment 5 (cg 2.5) placed at x=3, height 1:
    # H = 2*3 = 6, V = 2*1 + 5 = 7.  Add a mass-1 weight at x=-1, height 0:
    # M=3, H=5, V=7 -> R(0)=5/3, R(1)=4.
    rack = Rack.build("R", [{"name": "h", "x": "-1", "height": "0"}])
    place = [BottlePlacement.build("b", 3, 1, 2, 5)]
    sol = solve_rack(rack, place, [Weight("w", F(1))], [rep("c", 0, 1, 1, 4)])
    assert sol.base == {"mass": "2", "H": "6", "V": "7"}   # before any weight
    assert sol.feasible is True and sol.added_mass == 1
    assert sol.landing["c"]["landing_at_k_lo"] == str(F(5, 3))
    assert sol.landing["c"]["landing_at_k_hi"] == "4"


# ------------------------------------------------------------- validation guards
def test_too_many_holes_rejected():
    holes = [{"name": f"h{i}", "x": str(i), "height": "0"} for i in range(9)]
    with pytest.raises(RackError):
        Rack.build("R", holes)


def test_duplicate_hole_names_rejected():
    with pytest.raises(RackError):
        Rack.build("R", [{"name": "h", "x": "0"}, {"name": "h", "x": "1"}])


def test_bad_condition_intervals_rejected():
    with pytest.raises(RackError):
        Condition.build("c", 1, 0, -1, 1)
    with pytest.raises(RackError):
        Condition.build("c", 0, 1, 2, 1)


# --------------------------------------------------- manifest -> placements bridge
def test_placements_from_manifest():
    from prepquant.bottles import BottleType, manifest_from_loads

    b = BottleType.build("tall", [{"height": "8", "area": "1/2"}])
    manifest = manifest_from_loads({"RAW": (b, 2, 1)}).to_payload()
    from prepquant.bottles import read_manifest

    parsed = read_manifest(manifest)
    placements = placements_from_manifest(parsed, {"RAW": {"x": "0", "height": "0"}})
    assert len(placements) == 1
    assert placements[0].total_mass == 2
    assert placements[0].total_moment == 4


# -------------------------------- fill spanning two chambers drives the rack result
def test_cross_chamber_fill_two_holes_allocation_matters():
    """A bottle filled across two chambers (level 2, moment 3, mass 4) placed at
    the origin of a rack with a hole at x=-1 and one at x=+1.  With the narrow
    contact boundary [-1/4, 1/2] and k in [0,1] the correct hole is the LEFT one;
    the right one would make the landing worse.  This is the requested example in
    which a fill spanning two chambers plus two different hole assignments changes
    the conclusion."""
    from prepquant.bottles import BottleType, fill_bottle

    bottle = BottleType.build("cross", [{"height": "1", "area": "3"}, {"height": "2", "area": "1"}])
    fill = fill_bottle(bottle, wet_mass=4, wet_density=1)
    # first chamber volume 3 (level 1), then 1 of the 2 mL second chamber -> level 2
    assert fill.liquid_level == 2
    assert fill.liquid_moment == 3          # 3 kg@0.5 + 1 kg@1.5 = 1.5 + 1.5
    assert fill.total_mass == 4

    rack = Rack.build(
        "R",
        [{"name": "left", "x": "-1", "height": "0"}, {"name": "right", "x": "1", "height": "0"}],
    )
    placements = [BottlePlacement.build("B", 0, 0, fill.total_mass, fill.total_moment)]
    weights = [Weight("w", F(1))]
    cond = rep("c", 0, 1, F(-1, 4), F(1, 2))

    # base: R(0)=0, R(1)=3/4 -> fails the upper edge 1/2
    base = solve_rack(rack, placements, [], [cond])
    assert base.feasible is False

    # right hole: R(0)=1/5, R(1)=4/5 -> still fails
    # left hole:  R(0)=-1/5, R(1)=2/5 -> fits.  Only the left assignment works.
    sol = solve_rack(rack, placements, weights, [cond])
    assert sol.feasible is True
    assert sol.added_mass == 1
    assert sol.placements[0]["hole"] == "left"
    assert sol.landing["c"]["landing_at_k_lo"] == "-1/5"
    assert sol.landing["c"]["landing_at_k_hi"] == "2/5"


# ------------------------------------------- weight may only go in listed holes
def test_per_weight_allowed_holes():
    rack = Rack.build(
        "R",
        [{"name": "a", "x": "-1", "height": "0"}, {"name": "b", "x": "1", "height": "0"}],
    )
    placements = [BottlePlacement.build("B", 0, 0, 1, 2)]   # R=2 at k=1
    # the only usable weight is forbidden from 'a'; it must therefore use 'b',
    # which cannot fix the landing -> infeasible (no valid layout exists)
    weights = [Weight("w", F(1), allowed_holes=("b",))]
    sol = solve_rack(rack, placements, weights, [rep("c", 0, 1, -1, 1)])
    assert sol.feasible is False
    # allow 'a' and it becomes feasible, using 'a'
    weights2 = [Weight("w", F(1), allowed_holes=("a",))]
    sol2 = solve_rack(rack, placements, weights2, [rep("c", 0, 1, -1, 1)])
    assert sol2.feasible is True and sol2.placements[0]["hole"] == "a"


# ------------------------------------------- a finite set of historical racks
def test_solve_across_historical_racks_picks_cheapest():
    # rack A has a hole far out (needs a heavy weight); rack B has a hole at the
    # right spot (one light weight suffices).
    rack_a = Rack.build("A", [{"name": "far", "x": "-4", "height": "0"}])
    rack_b = Rack.build("B", [{"name": "near", "x": "-1", "height": "0"}])
    placements = [BottlePlacement.build("B", 0, 0, 1, 3)]   # base R=3 at k=1
    weights = [Weight("w", F(1))]
    result = solve_across_racks([rack_a, rack_b], placements, weights, [rep("c", 0, 1, -1, 1)])
    # rack B: H=-1, M=2, V=3 -> R(0)=-1/2, R(1)=1 -> fits with 1 kg
    # rack A: H=-4, M=2, V=3 -> R(0)=-2 out of [-1,1] -> infeasible
    assert result["chosen"] == "B"
    assert result["racks"]["B"]["feasible"] is True
    assert result["racks"]["B"]["added_mass"] == "1"
    assert result["racks"]["A"]["feasible"] is False

