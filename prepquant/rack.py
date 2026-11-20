"""Turnover-rack counterweight layout: minimum new mass over discrete holes.

Model
-----
A rack has fixed *bottle positions* and numbered *counterweight holes*.  Both
carry a horizontal position ``x`` and a height ``y`` above the rack face.  The
rack frame itself has a known mass and mass moment (both may be zero).

Each bottle load (from a bottle-load manifest) is placed whole at its position,
so a load of total mass ``M`` and first vertical moment ``Σm·z`` at position
``(x, y)`` contributes

    mass   M
    H      M·x
    V      M·y + (Σm·z)      # shift the vertical moment to the rack frame

The rack frame contributes its own ``(M0, H0, V0)``.  A candidate layout adds
numbered counterweights of given mass into *distinct* holes, each hole holding at
most one weight and each weight allowed only in its own permitted holes.

For a tilt parameter ``k`` the resultant's horizontal landing point is

    R(k) = (H_total + k·V_total) / M_total .

A condition is an interval of tilt ``k ∈ [L, R]`` together with a contact
boundary; the layout qualifies for that condition iff ``R(k)`` stays inside the
closed contact boundary ``[a, b]`` for *every* ``k`` in the interval.  Because
``R`` is linear in ``k`` this is exactly the two endpoint checks — and since all
conditions of one transport share the same layout, they must all hold at once.

One rack shares a single layout for the whole transport.  The task is to find the
layout of **minimum total added counterweight mass** that satisfies every
condition; if none does, that is reported honestly.  The search is exact: a
depth-first enumeration over (skip weight / put weight into an allowed free hole)
with objective pruning.  At most eight holes and eight weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

from .rational import ZERO, nonneg, q, qstr

MAX_HOLES = 8
MAX_WEIGHTS = 8


class RackError(Exception):
    pass


@dataclass
class Hole:
    name: str
    x: Fraction
    height: Fraction
    allowed_weight_masses: tuple[Fraction, ...] = ()

    def allows(self, mass: Fraction) -> bool:
        if not self.allowed_weight_masses:
            return True
        return mass in self.allowed_weight_masses


@dataclass
class Weight:
    name: str
    mass: Fraction
    allowed_holes: tuple[str, ...] = ()

    def allows_hole(self, hole_name: str) -> bool:
        if not self.allowed_holes:
            return True
        return hole_name in self.allowed_holes


@dataclass
class Rack:
    name: str
    frame_mass: Fraction = ZERO
    frame_moment_x: Fraction = ZERO   # H of the frame
    frame_moment_z: Fraction = ZERO   # V of the frame
    holes: tuple[Hole, ...] = ()

    @classmethod
    def build(
        cls,
        name: str,
        holes: Sequence[Mapping],
        frame_mass=0,
        frame_moment_x=0,
        frame_moment_z=0,
    ) -> "Rack":
        if len(holes) > MAX_HOLES:
            raise RackError(f"a rack holds at most {MAX_HOLES} holes, got {len(holes)}")
        parsed: list[Hole] = []
        seen: set[str] = set()
        for h in holes:
            hname = h.get("name")
            if not isinstance(hname, str) or not hname:
                raise RackError("each hole needs a non-empty name")
            if hname in seen:
                raise RackError(f"duplicate hole name {hname!r}")
            seen.add(hname)
            allowed = tuple(
                sorted({q(m, "allowed weight mass") for m in h.get("allowed_weights", ())})
            )
            parsed.append(
                Hole(
                    name=hname,
                    x=q(h["x"], f"hole {hname}.x"),
                    height=q(h.get("height", 0), f"hole {hname}.height"),
                    allowed_weight_masses=allowed,
                )
            )
        return cls(
            name=name,
            frame_mass=nonneg(q(frame_mass, "frame_mass"), "frame_mass"),
            frame_moment_x=q(frame_moment_x, "frame_moment_x"),
            frame_moment_z=q(frame_moment_z, "frame_moment_z"),
            holes=tuple(parsed),
        )


@dataclass
class Condition:
    name: str
    k_lo: Fraction
    k_hi: Fraction
    a: Fraction       # contact boundary lower edge
    b: Fraction       # contact boundary upper edge

    @classmethod
    def build(cls, name, k_lo, k_hi, a, b) -> "Condition":
        k_lo = q(k_lo, "k_lo")
        k_hi = q(k_hi, "k_hi")
        a = q(a, "a")
        b = q(b, "b")
        if k_lo > k_hi:
            raise RackError(f"condition {name!r}: tilt interval lower > upper")
        if a > b:
            raise RackError(f"condition {name!r}: contact boundary lower > upper")
        return cls(name=name, k_lo=k_lo, k_hi=k_hi, a=a, b=b)


@dataclass
class BottlePlacement:
    """A manifest load placed at a rack position."""

    position: str
    x: Fraction
    y: Fraction            # position height above the rack face
    total_mass: Fraction
    total_moment: Fraction  # Σm·z of the load about the bottle bottom

    @classmethod
    def build(cls, position, x, y, total_mass, total_moment) -> "BottlePlacement":
        return cls(
            position=position,
            x=q(x, "position.x"),
            y=q(y, "position.height"),
            total_mass=nonneg(q(total_mass, "total_mass"), "total_mass"),
            total_moment=q(total_moment, "total_moment"),
        )

    def contribution(self) -> tuple[Fraction, Fraction, Fraction]:
        m = self.total_mass
        h = m * self.x
        v = m * self.y + self.total_moment
        return m, h, v


def placements_from_manifest(manifest: Mapping, positions: Mapping[str, Mapping]) -> list[BottlePlacement]:
    """Bind each loaded bottle to a rack position ``{container: {x, height}}``."""
    out: list[BottlePlacement] = []
    for cid, entry in manifest["bottles"].items():
        if cid not in positions:
            raise RackError(f"no rack position given for loaded container {cid!r}")
        pos = positions[cid]
        load = entry["load"]
        out.append(
            BottlePlacement.build(
                position=cid,
                x=pos["x"],
                y=pos.get("height", 0),
                total_mass=load["total_mass"],
                total_moment=load["total_moment"],
            )
        )
    return out


def _landing(H: Fraction, V: Fraction, M: Fraction, k: Fraction) -> Fraction:
    return (H + k * V) / M


def _qualifies(
    H: Fraction,
    V: Fraction,
    M: Fraction,
    conditions: Sequence[Condition],
) -> bool:
    if M <= 0:
        return False
    for cond in conditions:
        for k in (cond.k_lo, cond.k_hi):
            landing = _landing(H, V, M, k)
            if landing < cond.a or landing > cond.b:
                return False
    return True


@dataclass
class RackSolution:
    feasible: bool
    added_mass: Fraction | None
    placements: list[dict] = field(default_factory=list)
    landing: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)
    reason: str | None = None
    explored_nodes: int = 0

    def to_payload(self) -> dict:
        return {
            "feasible": self.feasible,
            "added_mass": qstr(self.added_mass) if self.added_mass is not None else None,
            "placements": self.placements,
            "landing": self.landing,
            "base": self.base,
            "reason": self.reason,
            "explored_nodes": self.explored_nodes,
        }


def solve_rack(
    rack: Rack,
    placements: Sequence[BottlePlacement],
    weights: Sequence[Weight],
    conditions: Sequence[Condition],
) -> RackSolution:
    """Find the minimum-added-mass counterweight layout satisfying all conditions."""
    if len(rack.holes) > MAX_HOLES:
        raise RackError(f"rack has more than {MAX_HOLES} holes")
    if len(weights) > MAX_WEIGHTS:
        raise RackError(f"at most {MAX_WEIGHTS} counterweights are supported")
    if not conditions:
        raise RackError("at least one condition is required")

    # aggregate fixed mass / moments
    M0 = rack.frame_mass
    H0 = rack.frame_moment_x
    V0 = rack.frame_moment_z
    for p in placements:
        m, h, v = p.contribution()
        M0 += m
        H0 += h
        V0 += v
    base = {"mass": qstr(M0), "H": qstr(H0), "V": qstr(V0)}

    # Deterministic ordering: heaviest weights first so the moment is fixed fast
    # and the "skip" branch prunes effectively.
    wlist = sorted(weights, key=lambda w: (-w.mass, w.name))
    for w in wlist:
        if w.mass < 0:
            raise RackError(f"weight {w.name!r} has negative mass")
    # Allowed holes per weight: a hole accepts the weight's mass *and* the weight
    # lists that hole (empty list = anywhere).
    hole_allow: list[list[int]] = []
    for w in wlist:
        allowed = [
            j
            for j, h in enumerate(rack.holes)
            if h.allows(w.mass) and w.allows_hole(h.name)
        ]
        hole_allow.append(allowed)

    n = len(wlist)
    best_mass: list[Fraction | None] = [None]
    best_assign: list[tuple[int, int] | None] = [None]
    nodes = [0]

    # Symmetry breaking: two adjacent weights that are indistinguishable (same
    # mass, same allowed holes) need only be assigned in increasing hole order.
    identical_next = [
        wlist[i].mass == wlist[i + 1].mass and hole_allow[i] == hole_allow[i + 1]
        for i in range(n - 1)
    ]

    current: list[tuple[int, int] | None] = [None] * n

    def dfs(i: int, used: int, added: Fraction, Hw: Fraction, Vw: Fraction) -> None:
        nodes[0] += 1
        if best_mass[0] is not None and added >= best_mass[0]:
            return
        if i == n:
            if _qualifies(H0 + Hw, V0 + Vw, M0 + added, conditions):
                best_mass[0] = added
                best_assign[0] = tuple(current)
            return
        # skip this weight
        current[i] = None
        dfs(i + 1, used, added, Hw, Vw)
        # put it in an allowed free hole
        w = wlist[i]
        lower = -1
        if i > 0 and identical_next[i - 1] and current[i - 1] is not None:
            lower = current[i - 1][1]
        for j in hole_allow[i]:
            if used & (1 << j):
                continue
            if j <= lower:
                continue
            hole = rack.holes[j]
            current[i] = (i, j)
            dfs(
                i + 1,
                used | (1 << j),
                added + w.mass,
                Hw + w.mass * hole.x,
                Vw + w.mass * hole.height,
            )
        current[i] = None

    dfs(0, 0, ZERO, ZERO, ZERO)

    if best_mass[0] is None:
        return RackSolution(
            feasible=False,
            added_mass=None,
            base=base,
            reason="no counterweight layout satisfies every condition",
            explored_nodes=nodes[0],
        )

    assign = best_assign[0]
    placements_out = []
    Hw = ZERO
    Vw = ZERO
    for i, slot in enumerate(assign):
        if slot is None:
            continue
        _, j = slot
        w = wlist[i]
        hole = rack.holes[j]
        Hw += w.mass * hole.x
        Vw += w.mass * hole.height
        placements_out.append(
            {
                "weight": w.name,
                "mass": qstr(w.mass),
                "hole": hole.name,
                "x": qstr(hole.x),
                "height": qstr(hole.height),
            }
        )

    total_mass = M0 + best_mass[0]
    total_H = H0 + Hw
    total_V = V0 + Vw
    landing = _landing_report(total_H, total_V, total_mass, conditions)
    return RackSolution(
        feasible=True,
        added_mass=best_mass[0],
        placements=placements_out,
        landing=landing,
        base=base,
        explored_nodes=nodes[0],
    )


def _landing_report(H, V, M, conditions) -> dict:
    """Report R(k) at each condition's extreme parameters (the worst points)."""
    report = {}
    for cond in conditions:
        report[cond.name] = {
            "contact_boundary": [qstr(cond.a), qstr(cond.b)],
            "k_range": [qstr(cond.k_lo), qstr(cond.k_hi)],
            "landing_at_k_lo": qstr(_landing(H, V, M, cond.k_lo)),
            "landing_at_k_hi": qstr(_landing(H, V, M, cond.k_hi)),
        }
    return report


def solve_across_racks(
    racks: Sequence[Rack],
    placements: Sequence[BottlePlacement],
    weights: Sequence[Weight],
    conditions: Sequence[Condition],
) -> dict:
    """Try a finite set of historical racks and return the cheapest feasible one.

    Each rack is solved independently against the same transport conditions; the
    returned dict maps rack name -> :class:`RackSolution` payload, plus the
    ``chosen`` rack name (smallest added mass, ties broken by rack name) or
    ``None`` when no rack admits a layout.
    """
    results: dict[str, dict] = {}
    chosen = None
    chosen_mass = None
    for rack in sorted(racks, key=lambda r: r.name):
        sol = solve_rack(rack, placements, weights, conditions)
        results[rack.name] = sol.to_payload()
        if sol.feasible and (chosen is None or (sol.added_mass, rack.name) < (chosen_mass, chosen)):
            chosen = rack.name
            chosen_mass = sol.added_mass
    return {"chosen": chosen, "racks": results}
