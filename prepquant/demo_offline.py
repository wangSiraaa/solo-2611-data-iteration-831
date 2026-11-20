"""End-to-end demo of the offline retention model (two independent entry points).

Runs on fresh SQLite files under ``out/`` so it does not disturb the main demo:

1. a **retention batch** is replayed through the existing preparation ledger
   (build / transfer / mix / solvent / discard only) and its solute
   back-calculation is re-checked through the vault, proving the original
   quantitation still works;
2. the **wet-mass book** is replayed from the same ledger log, and two bottles of
   equal capacity and equal liquid wet mass — one tall, one wide — are filled;
3. a versioned **bottle-load manifest** is exported, then a *correction* of one
   bottle's type produces a successor manifest that still points at the previous
   version;
4. the **rack solver** consumes only the exported manifest and finds the minimum
   added counterweight mass for a set of transport conditions, reporting the
   landing at every extreme tilt parameter — before and after the correction;
5. the worked examples from the brief are reproduced (needs the only weight /
   widens to empty / corrected geometry needs none), plus a fill spanning two
   chambers with two holes whose allocation changes the conclusion.

Exits non-zero if any hand-checked value disagrees.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction as F
from pathlib import Path

from .bottles import BottleType, ManifestBuilder, fill_bottle, read_manifest
from .ledger import Ledger
from .rack import BottlePlacement, Condition, Rack, Weight, placements_from_manifest, solve_rack
from .vault import Vault
from .wetmass import WetMassBook

OUT = Path(__file__).resolve().parent.parent / "out" / "offline"


class Failure(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Failure(msg)


def _retention_batch() -> list[dict]:
    """A batch restricted to build / transfer / mix / solvent / discard.

    RAW holds one sample; two equal draws feed A and B; a small aliquot of A is
    mixed with solvent into C.
    """
    return [
        {"op": "sample", "container": "RAW", "volume": "6", "sample": "s1"},
        {"op": "transfer", "source": "RAW", "volume": "3", "destination": "A"},
        {"op": "transfer", "source": "RAW", "volume": "3", "destination": "B"},
        {"op": "solvent", "container": "A", "volume": "1"},
        {"op": "mix", "sources": [{"container": "B", "volume": "1"}], "destination": "C"},
        {"op": "discard", "container": "C"},
    ]


def _solute_backcalc(L: Ledger, V: Vault) -> dict:
    """Re-check the original solute quantitation still works on this batch."""
    payload = L.export_product("A")
    V.add_measurement("offline-r", payload, "6", "1")
    res = V.analyze("offline-solute", ["offline-r"], {"s1": ["0", "100"]})
    # A holds 3 mL of RAW in 4 mL total -> mass 3*s1; reading 6±1 -> s1 in [20/3, 28/3]
    return res


def offline_demo() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report: dict = {"checks": [], "manifests": {}}

    # -- 1. ledger batch + solute back-calculation ------------------------
    L = Ledger(OUT / "offline.ledger.db")
    V = Vault(OUT / "offline.vault.db")
    L.apply_batch(_retention_batch())
    solute = _solute_backcalc(L, V)
    check(solute["status"] == "feasible", "solute back-calculation must stay feasible")
    check(
        solute["endpoints"]["s1"]["min"] == "20/3" and solute["endpoints"]["s1"]["max"] == "28/3",
        f"solute endpoints expected [20/3,28/3], got {solute['endpoints']}",
    )
    report["checks"].append("original solute back-calculation still works on the retention batch")

    # -- 2. wet-mass replay from the same ledger log ----------------------
    rho = "3/2"                       # 1.5 wet mass units per mL
    book = WetMassBook.from_ledger(L, {"RAW": rho}, "1")
    # RAW: 6 mL -> wet 9; two 3 mL draws -> wet 9/2 each; then 1 mL leaves B for C,
    # and C is discarded.  B keeps 2 mL -> wet 3.
    check(book.wet_mass("A") == F(9, 2) + 1, f"A wet mass unexpected: {book.wet_mass('A')}")
    check(book.wet_mass("B") == F(3), f"B wet mass unexpected: {book.wet_mass('B')}")
    check(book.wet_mass("C") == 0, "C was discarded, wet mass must be zero")
    report["checks"].append("wet mass replayed from the ledger: proportional transfers, solvent added, discard zeroes")

    # Two bottles, same capacity 4 and same liquid wet mass 3, different shapes.
    tall = BottleType.build("tall", [{"height": "8", "area": "1/2"}], shell_mass=0, shell_cg_height=0)
    wide = BottleType.build("wide", [{"height": "4", "area": "1"}], shell_mass=0, shell_cg_height=0)
    check(tall.total_capacity() == 4 and wide.total_capacity() == 4, "bottle capacities must match")
    wet = book.wet_mass("B")         # 3
    fill_tall = fill_bottle(tall, wet, rho)   # volume 2
    fill_wide = fill_bottle(wide, wet, rho)
    check(fill_tall.total_mass == fill_wide.total_mass == 3, "same liquid wet mass in both bottles")
    check(fill_tall.liquid_level == 4 and fill_wide.liquid_level == 2,
          f"levels expected 4/2, got {fill_tall.liquid_level}/{fill_wide.liquid_level}")
    check(fill_tall.liquid_moment == 6 and fill_wide.liquid_moment == 3,
          f"moments expected 6/3, got {fill_tall.liquid_moment}/{fill_wide.liquid_moment}")
    report["checks"].append("two equal-capacity bottles: tall has level 4 / moment 6, wide level 2 / moment 3")

    # -- 3. versioned manifest + correction -------------------------------
    builder = ManifestBuilder()
    builder.add("A", tall, wet, rho)
    builder.add("B", tall, wet, rho)
    v1 = builder.build(note="initial tall layout")
    v1.save(OUT / "manifest_v1.json")

    # correct bottle B to the wide type
    b2 = ManifestBuilder()
    v2 = b2.revise(v1, "B", wide, wet, rho, note="Bottle B corrected to wide geometry")
    v2.save(OUT / "manifest_v2.json")
    check(v2.version == v1.version + 1, "correction must bump the manifest version")
    check(v2.previous == {"version": 1, "manifest_hash": v1.manifest_hash},
          "successor must point at the previous manifest hash")
    # the previous layout is still readable on its own
    reread_v1 = read_manifest(json.loads((OUT / "manifest_v1.json").read_text()))
    check(reread_v1["bottles"]["B"]["load"]["total_moment"] == 6,
          "previous manifest must remain independently readable")
    report["manifests"]["v1"] = v1.to_payload()
    report["manifests"]["v2"] = v2.to_payload()
    report["checks"].append("bottle-type correction produced manifest v2 linked to v1 and v1 still re-readable")

    # -- 4. rack solve from the exported manifests ------------------------
    rack = Rack.build(
        "shelf-1",
        [{"name": "far", "x": "-2", "height": "0"}],
        frame_mass=0,
    )
    conditions = [Condition.build("tilt", "0", "1", "-1", "1")]
    weights = [Weight("cw-1", F(1))]

    def solve_with(manifest):
        parsed = read_manifest(manifest.to_payload())
        placements = placements_from_manifest(parsed, {"A": {"x": "0", "height": "0"},
                                                        "B": {"x": "0", "height": "0"}})
        return solve_rack(rack, placements, weights, conditions)

    before = solve_with(v1)   # both tall: base V = 12, M = 6 -> R(1)=2, too far right
    after = solve_with(v2)    # A tall, B wide: V = 6 + 3 = 9, M = 6 -> R(1)=3/2
    report["rack_before"] = before.to_payload()
    report["rack_after"] = after.to_payload()
    check(before.feasible is False, "both-tall layout should be infeasible on the narrow boundary")
    check(after.feasible is True and after.added_mass == 1,
          f"A tall + B wide should need 1 unit, got {after.added_mass}")
    check(after.placements[0]["hole"] == "far", "the far hole is the feasible choice after correction")
    report["checks"].append("rack solve reads only the exported manifest; shape correction changes the verdict")

    # -- 5a. worked example: needs the only weight, worst landing 1 -------
    ex_rack = Rack.build("ex", [{"name": "h", "x": "0", "height": "0", "allowed_weights": ["2"]}])
    ex_place = [BottlePlacement.build("b", 0, 0, 2, 4)]
    ex_w = [Weight("w", F(2))]
    ex = solve_rack(ex_rack, ex_place, ex_w, [Condition.build("c", 0, 1, -1, 1)])
    check(ex.feasible and ex.added_mass == 2 and ex.landing["c"]["landing_at_k_hi"] == "1",
          "worked example: must use the 2-unit weight, worst landing 1")
    widened = solve_rack(ex_rack, ex_place, ex_w, [Condition.build("c", 0, 1, -2, 2)])
    check(widened.feasible and widened.added_mass == 0,
          "worked example: widening the boundary empties the hole")
    corrected = solve_rack(ex_rack, [BottlePlacement.build("b", 0, 0, 2, 2)], ex_w,
                           [Condition.build("c", 0, 1, -1, 1)])
    check(corrected.feasible and corrected.added_mass == 0,
          "worked example: corrected area-1 bottle needs no weight")
    report["worked_example"] = {"narrow": ex.to_payload(), "widened": widened.to_payload(),
                                "corrected": corrected.to_payload()}
    report["checks"].append("worked example reproduced: needs weight / widened no weight / corrected no weight")

    # -- 5b. cross-chamber fill + two holes changes the conclusion --------
    cross = BottleType.build("cross", [{"height": "1", "area": "3"}, {"height": "2", "area": "1"}])
    cross_fill = fill_bottle(cross, wet_mass=4, wet_density=1)
    check(cross_fill.liquid_level == 2 and cross_fill.liquid_moment == 3,
          "cross-chamber fill must end at level 2 with moment 3")
    cross_rack = Rack.build(
        "cross-rack",
        [{"name": "left", "x": "-1", "height": "0"}, {"name": "right", "x": "1", "height": "0"}],
    )
    cross_place = [BottlePlacement.build("b", 0, 0, cross_fill.total_mass, cross_fill.total_moment)]
    cross_cond = [Condition.build("c", 0, 1, "-1/4", "1/2")]
    cross_sol = solve_rack(cross_rack, cross_place, [Weight("cw", F(1))], cross_cond)
    check(cross_sol.feasible and cross_sol.placements[0]["hole"] == "left",
          "cross-chamber example must require the LEFT hole")
    # the right hole would leave the landing outside the boundary
    right_only = solve_rack(
        Rack.build("r", [{"name": "right", "x": "1", "height": "0"}]),
        cross_place, [Weight("cw", F(1))], cross_cond,
    )
    check(right_only.feasible is False, "the right hole must be infeasible, so allocation matters")
    report["cross_chamber"] = {
        "fill": cross_fill.to_payload(),
        "solution": cross_sol.to_payload(),
        "right_hole_only_feasible": right_only.feasible,
    }
    report["checks"].append("fill spanning two chambers + two holes: allocation changes the conclusion")

    L.close()
    V.close()
    return report


def main() -> int:
    try:
        report = offline_demo()
    except Failure as exc:
        print(f"OFFLINE DEMO FAILED: {exc}")
        return 1
    (OUT / "offline_demo_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("offline demo OK")
    for c in report["checks"]:
        print(f"  - {c}")
    print(f"report written to {OUT / 'offline_demo_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
